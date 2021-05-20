from copy import deepcopy
import sys
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

import models
from utils.misc import update_after_resize
from utils.heatmap_decoding import get_final_preds
from utils.misc import DLT_pytorch, DLT, triangulate_ransac, DLT_sii_pytorch
from .triangulation_model_utils import  op, multiview, volumetric
from . import pose_hrnet_softmax, pose_hrnet_volumetric, pose_hrnet, CPM_volumetric
from .v2v import V2VModel

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        # 3D pose + KCS + KCST
        input_dim = config.DATASET.NUM_JOINTS * 3 + 800
        self.reduce = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        linear_lst = []
        for i in range(4):
            linear_lst.append(nn.Linear(128, 128)),
            linear_lst.append(nn.LeakyReLU(0.2, inplace=True))
        self.linear = nn.Sequential(*linear_lst)

        self.head = nn.Linear(128, 1)

    def forward(self, x):
        # x is the concatenation of 3D pose, KCS and TKCS
        x1 = self.reduce(x)
        x2 = self.linear(x1) + x1
        x3 = self.head(x2)
        return x3

class RANSACTriangulationNet(nn.Module):
    def __init__(self, config, is_train=True):
        super().__init__()
        self.orig_img_size = [640,480]
        self.backbone = eval(config.MODEL.BACKBONE_NAME + '.get_pose_net(config, is_train=True)')
        backbone_path = config.MODEL.BACKBONE_MODEL_PATH
        if backbone_path:
            checkpoint = torch.load(backbone_path, map_location='cpu')
            if 'state_dict' in checkpoint.keys():
                state_dict = checkpoint['state_dict']
                print("=> Loading pretrained {} backbone from '{}' (epoch {})".format(
                    config.MODEL.BACKBONE_NAME, backbone_path, checkpoint['epoch']))
            else:
                state_dict = checkpoint
                print("=> Loading pretrained {} backbone from '{}'".format(
                    config.MODEL.BACKBONE_NAME, backbone_path))

            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)
            
            self.backbone.load_state_dict(state_dict, strict=False)

        self.direct_optimization = config.MODEL.DIRECT_OPTIMIZATION
        self.heatmap_softmax = config.MODEL.HEATMAP_SOFTMAX

    def forward(self, images, proj_matrices):
        batch_size, n_views = images.shape[:2]
        orig_width, orid_height = self.orig_img_size
        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integrate
        heatmaps, _, _, _ = self.backbone(images) # 4 x 21 x 64 x 64
        
        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        n_joints, heatmap_shape = heatmaps.shape[1], tuple(heatmaps.shape[2:])

        keypoints_2d = get_final_preds(heatmaps, self.heatmap_softmax).view(batch_size, n_views, -1, 2)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d).float()
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (orig_width / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (orid_height / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # triangulate (cpu)
        keypoints_2d_np = keypoints_2d.detach().cpu().numpy()
        proj_matricies_np = proj_matrices.detach().cpu().numpy()

        # contrast 21 x 3
        #keypoint_3d_in_base_camera_test = np.stack([DLT(keypoints_2d_np[b], proj_matricies_np[b]) for b in range(batch_size)])
        #keypoint_3d_in_base_camera_test = np.stack([multiview.triangulate_point_from_multiple_views_linear(proj_matrices[0], keypoints_2d_np[0,:,k]) for k in range(21)])
        #print(keypoint_3d_in_base_camera_test)

        keypoints_3d = np.zeros((batch_size, n_joints, 3))
        confidences = np.zeros((batch_size, n_views, n_joints))  # plug
        for batch_i in range(batch_size):
            for joint_i in range(n_joints):
                current_proj_matricies = proj_matricies_np[batch_i]
                points = keypoints_2d_np[batch_i, :, joint_i]
                keypoint_3d, _ = triangulate_ransac(current_proj_matricies, points, reprojection_error_epsilon=25, direct_optimization=self.direct_optimization)
                keypoints_3d[batch_i, joint_i] = keypoint_3d

        keypoints_3d = torch.from_numpy(keypoints_3d).type(torch.float).to(images.device)
        confidences = torch.from_numpy(confidences).type(torch.float).to(images.device)

        return keypoints_3d, keypoints_2d, heatmaps, confidences

    def triangulate_ransac(self, proj_matrices, points, n_iters=10, reprojection_error_epsilon=15, direct_optimization=True):
        # proj_matrices: N_views x 3 x 4
        # points: N_views x 2
        assert len(proj_matrices) == len(points)
        assert len(points) >= 2

        proj_matrices = np.array(proj_matrices)
        points = np.array(points)

        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()
        for i in range(n_iters):
            sampled_views = sorted(random.sample(view_set, 2)) # get 2 views randomly

            # recover 3D world coordinates (size: (3,)) by using DLT
            keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(proj_matrices[sampled_views], points[sampled_views])
            # calculate the distance between the groundtruth and the reprojected 2D pose (size: N_views)
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), points, proj_matrices)[0]

            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]
                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matrices[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points)
        reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)

        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization (using least square to refine the DLT result)
        if direct_optimization:
            def residual_function(x):
                reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera) # initial guess
            res = least_squares(residual_function, x_0, loss='huber', method='trf')

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)

        return keypoint_3d_in_base_camera, inlier_list


class AlgebraicTriangulationNet(nn.Module):
    def __init__(self, config, is_train=True):
        super().__init__()
        print("=> Initializing AlgebraicTriangulationNet...")
        self.heatmap_softmax = config.MODEL.HEATMAP_SOFTMAX
        self.use_alg_confidences = config.MODEL.ALG_CONFIDENCES

        self.backbone = eval(config.MODEL.BACKBONE_NAME + '.get_pose_net(config, is_train=True)')
        backbone_path = config.MODEL.BACKBONE_MODEL_PATH
        if backbone_path:
            checkpoint = torch.load(backbone_path, map_location='cpu')
            if 'state_dict' in checkpoint.keys():
                state_dict = checkpoint['state_dict']
                print("=> Loading pretrained {} backbone from '{}' (epoch {})".format(
                    config.MODEL.BACKBONE_NAME, backbone_path, checkpoint['epoch']))
            else:
                state_dict = checkpoint
                print("=> Loading pretrained {} backbone from '{}'".format(
                    config.MODEL.BACKBONE_NAME, backbone_path))

            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)
            
            self.backbone.load_state_dict(state_dict, strict=False)

        # freeze lower layers
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.stage4.parameters():
            p.requires_grad = True
        for p in self.backbone.last_layer.parameters():
            p.requires_grad = True

    def forward(self, images, proj_matrices=torch.rand((1,4,3,4)), orig_img_size=[640,480]):
        # images: b x v x 3 x 256(H) x 256(W)
        # proj_matrices: b x v x 3 x 4
        # orig_img_size: [W,H]
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:]) # b*v x 3 x H x W

        # forward backbone and integral
        if self.use_alg_confidences:
            # alg_confidences: (b*N_views) x N_joints
            heatmaps, _, alg_confidences, _ = self.backbone(images)
            alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:]) # b x N_views x N_joints
            # norm confidences
            alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True) + 1e-5 # for numerical stability
        else:
            heatmaps, _, _, _ = self.backbone(images)
            alg_confidences = None

        keypoints_2d = get_final_preds(heatmaps, use_softmax=self.heatmap_softmax) # b*v x 21 x 2

        # reshape back
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:]) # b x v x 21 x 2
        
        # upscale keypoints_2d so that it is located in the original image
        heatmap_size = heatmaps.shape[-1]
        keypoints_2d[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (orig_img_size[0] / heatmap_size) # u
        keypoints_2d[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (orig_img_size[1] / heatmap_size) # v
        # keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        # keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (images.shape[-1] / heatmaps.shape[-1]) # u
        # keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (images.shape[-2] / heatmaps.shape[-2]) # v
        # keypoints_2d = keypoints_2d_transformed

        # triangulate
        try:
            if self.use_alg_confidences:
                keypoints_3d = multiview.triangulate_batch_of_points(
                    proj_matrices, keypoints_2d,
                    confidences_batch=alg_confidences
                )
            else:
                keypoints_3d = torch.cat(
                    [DLT_sii_pytorch(keypoints_2d[:,:,k], proj_matrices).unsqueeze(1) for k in range(keypoints_2d.shape[2])],
                    dim=1
                ) # b x 21 x 3

        except RuntimeError as e:
            print("Error: ", e)

            print("confidences =", confidences_batch_pred)
            print("proj_matrices = ", proj_matrices)
            print("keypoints_2d_batch_pred =", keypoints_2d_batch_pred)
            exit()

        return keypoints_3d, keypoints_2d, heatmaps, alg_confidences


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, is_train=True):
        super().__init__()
        self.config = config
        self.num_joints = config.DATASET.NUM_JOINTS
        self.volume_aggregation_method = config.MODEL.VOLUME_AGGREGATION_METHOD

        # volume
        self.volume_softmax = config.MODEL.VOLUME_SOFTMAX
        self.volume_multiplier = config.MODEL.VOLUME_MULTIPLIER
        self.volume_size = config.MODEL.VOLUME_SIZE

        self.cuboid_side = config.MODEL.CUBOID_SIZE

        # self.kind = config.model.kind
        self.heatmap_softmax = config.MODEL.HEATMAP_SOFTMAX
        self.use_gt_middleroot = config.MODEL.USE_GT_MIDDLEROOT

        # heatmap
        # self.heatmap_softmax = config.model.heatmap_softmax
        # self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        #self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        # config.MODEL.ALG_CONFIDENCES = False
        # config.MODEL.VOL_CONFIDENCES = False
        # if self.volume_aggregation_method.startswith('conf'):
        #     config.MODEL.VOL_CONFIDENCES = True
        self.orig_img_size=[640,480]
        self.backbone = eval(config.MODEL.BACKBONE_NAME + '.get_pose_net(config, is_train=True)')
        #self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        if is_train:
            backbone_path = config.MODEL.BACKBONE_MODEL_PATH
            if backbone_path:
                checkpoint = torch.load(backbone_path, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    state_dict = checkpoint['state_dict']
                    print("=> Loading pretrained {} backbone from '{}' (epoch {})".format(
                        config.MODEL.BACKBONE_NAME, backbone_path, checkpoint['epoch']))
                else:
                    state_dict = checkpoint
                    print("=> Loading pretrained {} backbone from '{}'".format(
                        config.MODEL.BACKBONE_NAME, backbone_path))

                for key in list(state_dict.keys()):
                    new_key = key.replace("module.", "")
                    state_dict[new_key] = state_dict.pop(key)
                
                self.backbone.load_state_dict(state_dict, strict=False)
                
            # freeze lower layers
            if 'hrnet' in config.MODEL.BACKBONE_NAME:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for p in self.backbone.stage3.parameters():
                    p.requires_grad = False
                for p in self.backbone.stage4.parameters():
                    p.requires_grad = True
                for p in self.backbone.last_layer.parameters():
                    p.requires_grad = True
                try:
                    self.backbone.trainable_temp.requires_grad = False
                except:
                    print('No trainable temperature')
                    pass

        self.process_features = nn.Sequential(
            nn.Conv2d(sum(config.MODEL.EXTRA.STAGE4.NUM_CHANNELS), 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)


    def forward(self, images, proj_matrices=torch.rand((1,4,3,4)), batch=None, keypoints_3d=None):
        # images: b x N_views x 3 x H x W
        # proj_matricies_batch (K*H): b x N_views x 3 x 4
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        # heatmaps: (b*N_views) x 21 x 64 x 64
        # features: (b*N_views) x 480 x 64 x 64
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # find the middle finger root position
        base_idx = 9
        # v2: 3D EPE 10.7802
        pose2d_pred = get_final_preds(heatmaps, use_softmax=self.heatmap_softmax).view(batch_size, n_views, heatmaps.shape[1], 2) # batch_size x N_views x 21 x 2
        base_points = torch.cat([DLT_pytorch(pose2d_pred[b,:,9:10], proj_matrices[b]) for b in range(batch_size)], dim=0)

        # V3: 
        # pose2d_pred_temp = get_final_preds(heatmaps, use_softmax=self.heatmap_softmax).view(batch_size, n_views, heatmaps.shape[1], 2) # batch_size x N_views x 21 x 2
        # pose2d_pred = torch.zeros(pose2d_pred_temp.shape, dtype=pose2d_pred_temp.dtype, device=pose2d_pred_temp.device)
        # orig_width, orig_height = self.orig_img_size
        # pose2d_pred[:,:,:,0] = pose2d_pred_temp[:,:,:,0] * orig_width / 64
        # pose2d_pred[:,:,:,1] = pose2d_pred_temp[:,:,:,1] * orig_height / 64
        # base_points = DLT_sii_pytorch(pose2d_pred[:,:,base_idx], proj_matrices) # b x 3

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # camera intrinsics already changed in function3D.py

        # new_cameras = deepcopy(batch['cameras'])
        # for view_i in range(n_views):
        #     for batch_i in range(batch_size):
        #         new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        #proj_matrices = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        #proj_matrices = proj_matrices.float().to(device)

        # build coord volumes
        cuboids = []
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            # if self.use_gt_middleroot:
            #     keypoints_3d = batch['keypoints_3d'][batch_i]
            # else:
            base_point = base_points[batch_i]

            # build cuboid
            sides = torch.tensor([self.cuboid_side, self.cuboid_side, self.cuboid_side], device=base_points.device) # 2500 x 2500 x 2500 (mm)
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume, dividing the cubic length (2500mm) into 63 segments. NOTE: meshgrid returns a tuple
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float) # 64 x 64 x 64 x 3
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            # self.volume_size - 1 because we fill the cube with the global coords of each voxel's center
            # the elements of the grid are bound in [0,63)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            axis = [0, 1, 0]  # y axis

            # rotate
            coord_volume = coord_volume - base_point
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + base_point

            # transfer
            # if self.transfer_cmu_to_human36m:  # different world coordinates
            #     coord_volume = coord_volume.permute(0, 2, 1, 3)
            #     inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
            #     coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume # batch_size x 64 x 64 x 64 x 3

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features) # 32 output channels
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume: b x 32 x 64 x 64 x 64
        volumes = op.unproject_heatmaps(features, proj_matrices, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, pose2d_pred, heatmaps, volumes, vol_confidences, coord_volumes, base_points

class VolumetricTriangulationNet_CPM(nn.Module):
    def __init__(self, config, is_train=True):
        super().__init__()
        self.config = config
        self.orig_img_size = [640,480]
        self.num_joints = config.DATASET.NUM_JOINTS
        self.volume_aggregation_method = config.MODEL.VOLUME_AGGREGATION_METHOD

        # volume
        self.volume_softmax = config.MODEL.VOLUME_SOFTMAX
        self.volume_multiplier = config.MODEL.VOLUME_MULTIPLIER
        self.volume_size = config.MODEL.VOLUME_SIZE

        self.cuboid_side = config.MODEL.CUBOID_SIZE

        # self.kind = config.model.kind
        self.heatmap_softmax = config.MODEL.HEATMAP_SOFTMAX
        self.use_gt_middleroot = config.MODEL.USE_GT_MIDDLEROOT

        self.backbone = CPM_volumetric.get_pose_net(config, is_train=True)
        #self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        if is_train:
            backbone_path = config.MODEL.BACKBONE_MODEL_PATH
            if backbone_path:
                checkpoint = torch.load(backbone_path, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    state_dict = checkpoint['state_dict']
                    print("=> Loading pretrained {} backbone from '{}' (epoch {})".format(
                        config.MODEL.BACKBONE_NAME, backbone_path, checkpoint['epoch']))
                else:
                    state_dict = checkpoint
                    print("=> Loading pretrained {} backbone from '{}'".format(
                        config.MODEL.BACKBONE_NAME, backbone_path))

                for key in list(state_dict.keys()):
                    new_key = key.replace("module.", "")
                    state_dict[new_key] = state_dict.pop(key)
                
                self.backbone.load_state_dict(state_dict, strict=False)
                
            # freeze lower layers
            for p in self.backbone.parameters():
                p.requires_grad = False

            # for p in self.backbone.conv1_stage6.parameters():
            #     p.requires_grad = True
            # for p in self.backbone.Mconv1_stage6.parameters():
            #     p.requires_grad = True
            # for p in self.backbone.Mconv2_stage6.parameters():
            #     p.requires_grad = True
            for p in self.backbone.Mconv3_stage6.parameters():
                p.requires_grad = True
            for p in self.backbone.Mconv4_stage6.parameters():
                p.requires_grad = True
            for p in self.backbone.Mconv5_stage6.parameters():
                p.requires_grad = True


        self.process_features = nn.Sequential(
            nn.Conv2d(128, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)


    def forward(self, images, centermaps=torch.rand(1,4,1,256,256), proj_matrices=torch.rand((1,4,3,4)), batch=None, keypoints_3d=None):
        # images: b x N_views x 3 x H x W
        # proj_matricies_batch (K*H): b x N_views x 3 x 4
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])
        centermaps = centermaps.view(-1, *centermaps.shape[2:]) # b x 1 x 256 x 256
        # forward backbone
        # heatmaps: (b*N_views) x 22 x 64 x 64
        # features: (b*N_views) x 55 x 64 x 64
        # vol_confidences: b x 32
        _,_,_,_,_, heatmaps, features, vol_confidences = self.backbone(images, centermaps)
        # find the middle finger root position
        base_idx = 9
        # v2: 3D EPE 10.7802
        pose2d_pred = get_final_preds(heatmaps[:,1:], use_softmax=self.heatmap_softmax).view(batch_size, n_views, heatmaps.shape[1] - 1, 2) # batch_size x N_views x 21 x 2
        base_points = torch.cat([DLT_pytorch(pose2d_pred[b,:,base_idx:base_idx+1], proj_matrices[b]) for b in range(batch_size)], dim=0)

        # V3: 
        # pose2d_pred_temp = get_final_preds(heatmaps[:,1:], use_softmax=self.heatmap_softmax).view(batch_size, n_views, heatmaps.shape[1] - 1, 2) # batch_size x N_views x 21 x 2
        # pose2d_pred = torch.zeros(pose2d_pred_temp.shape, dtype=pose2d_pred_temp.dtype, device=pose2d_pred_temp.device)
        # orig_width, orig_height = self.orig_img_size
        # pose2d_pred[:,:,:,0] = pose2d_pred_temp[:,:,:,0] * orig_width / 64
        # pose2d_pred[:,:,:,1] = pose2d_pred_temp[:,:,:,1] * orig_height / 64
        # base_points = DLT_sii_pytorch(pose2d_pred[:,:,base_idx], proj_matrices) # b x 3

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # camera intrinsics already changed in function3D.py

        # new_cameras = deepcopy(batch['cameras'])
        # for view_i in range(n_views):
        #     for batch_i in range(batch_size):
        #         new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        #proj_matrices = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        #proj_matrices = proj_matrices.float().to(device)

        # build coord volumes
        cuboids = []
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            # if self.use_gt_middleroot:
            #     keypoints_3d = batch['keypoints_3d'][batch_i]
            # else:
            base_point = base_points[batch_i]

            # build cuboid
            sides = torch.tensor([self.cuboid_side, self.cuboid_side, self.cuboid_side], device=base_points.device) # 2500 x 2500 x 2500 (mm)
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume, dividing the cubic length (2500mm) into 63 segments. NOTE: meshgrid returns a tuple
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float) # 64 x 64 x 64 x 3
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            # self.volume_size - 1 because we fill the cube with the global coords of each voxel's center
            # the elements of the grid are bound in [0,63)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            axis = [0, 1, 0]  # y axis

            # rotate
            coord_volume = coord_volume - base_point
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + base_point

            # transfer
            # if self.transfer_cmu_to_human36m:  # different world coordinates
            #     coord_volume = coord_volume.permute(0, 2, 1, 3)
            #     inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
            #     coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume # batch_size x 64 x 64 x 64 x 3

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features) # 32 output channels
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume: b x 32 x 64 x 64 x 64
        volumes = op.unproject_heatmaps(features, proj_matrices, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, pose2d_pred, heatmaps, volumes, vol_confidences, coord_volumes, base_points
