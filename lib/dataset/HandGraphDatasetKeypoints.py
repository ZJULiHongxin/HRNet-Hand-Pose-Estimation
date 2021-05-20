import os
import shutil
import time

from .HandGraphDataset import HandGraphDataset

import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .HandGraph_utils.utils import *
from .HandGraph_utils.vis import *
from dataset.frei_utils.fh_utils import *

class HandGraphDataset_Keypoint(HandGraphDataset):
    def __init__(self, config, set_name, heatmap_generator, transforms=None):
        r"""Read RGB images and their ground truth heat maps
        params:
        @ set_name: the name of the training or validation data directory
        @ mode: indicates whether the this is a training set ot a test set
        @ transform: data augmentation is implemented if transformers are given
        """
        super().__init__(config.DATA_DIR,
                        set_name,
                        config.DATASET.DATA_FORMAT)

        assert config.DATASET.NUM_JOINTS == 21, 'Number of joint for RHD is 21!'
        self.set_name = set_name
        self.config = config
        self.num_joints = config.DATASET.NUM_JOINTS
        self.image_size = np.array(config.MODEL.IMAGE_SIZE)
        self.heatmap_generator = heatmap_generator
        self.transforms = transforms
        self.scale_factor = config.DATASET.SCALE_FACTOR
        self.rotation_factor = config.DATASET.ROT_FACTOR
        self.flip = config.DATASET.FLIP

    def __len__(self):
        return super().__len__()
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.set_name)
        return fmt_str

    def generate_heat_maps(self, joint_x, joint_y, visible, hm_size):
        gaussian_blur_kernel = kornia.filters.GaussianBlur2d((self.config.GAUSSIAN_KERNEL, self.config.GAUSSIAN_KERNEL),
                                                             (self.config.DATASET.SIGMA, self.config.DATASET.SIGMA),
                                                             border_type='constant')
        hm_lst = []

        for kp_idx in range(21):
            if visible[kp_idx]:
                hm = torch.zeros((1,1,hm_size, hm_size))

                col = min(63,max(0,int(joint_x[kp_idx])))
                row = min(63,max(0,int(joint_y[kp_idx])))

                hm[0][0][row][col] = 1
                normalized_hm = gaussian_blur_kernel(hm).squeeze()
            else:
                normalized_hm = torch.zeros((hm_size, hm_size))
            hm_lst.append(normalized_hm)
        
        return hm_lst

    def generate_heat_maps_without_quantization(self, joint_x, joint_y, visible, hm_size):
        """
        params:
        @ joint_x: u-axis (rightwards)
        @ joint_y: v-axis (downwards)
        """
        hm_lst = []
        sigma = self.sigma

        for kp_idx in range(21):
            if visible[kp_idx]:
                hm = torch.zeros((hm_size, hm_size), dtype=torch.float32)

                col = min(hm_size - 1.0,max(0.0, joint_x[kp_idx]))
                row = min(hm_size - 1.0,max(0.0, joint_y[kp_idx]))
                row_idx, col_idx = 0, 0
                for i in range(0, 6 * sigma + 1):
                    row_idx = int(row) - 3 * sigma + i
                    if 0 <= row_idx < hm_size:
                        for j in range(0, 6 * sigma + 1):
                            col_idx = int(col) - 3 * sigma + j
                            if 0 <= col_idx < hm_size:
                                d = (row_idx - row) ** 2 + (col_idx - col) ** 2
                                if d < 16 * sigma * sigma:
                                    hm[row_idx][col_idx] = torch.exp(-1 * torch.tensor(d, dtype=torch.float32) / (2 * sigma * sigma))
                                    
                normalized_hm = hm / hm.sum()
            else: 
                normalized_hm = torch.zeros((hm_size, hm_size))

            hm_lst.append(normalized_hm)
        return hm_lst

    def __getitem__(self, idx):
        #ori_img, img_mask, pose2d, local_pose3d_gt, mesh_2d, local_mesh_pts_gt, mesh_tri_idx, cam_proj_mat, img_path = super().__getitem__(idx)
        ori_img, pose2d, img_path = super().__getitem__(idx)
        #pose_mesh_2d = np.vstack((pose2d, mesh_2d)) # (21+N) x 3

        # for training and validation; No data augmentation yet !
        # img: torch.tensor of size 3 x 256 x 256; # pose_mesh_2d is a list containing only one element which is a 
        # np.array of size (21+N)x3. The 3rd column stands for visibility
        img, pose2d = self.transforms( 
            ori_img, [pose2d]         
        )

        #pose2d = pose_mesh_2d[0][0:pose2d.shape[0], :]
        #mesh_2d = pose_mesh_2d[0][pose2d.shape[0]:, 0:-1]
        pose2d = pose2d[0]
        # z = np.vstack((np.expand_dims(local_pose3d_gt[:,2], axis=1), # (21+N) x 1
        #                 np.expand_dims(local_mesh_pts_gt[:, 2], axis=1)))
        
        # 3D annotations 
        # factor = self.config.MODEL.IMAGE_SIZE[0] / self.config.MODEL.HEATMAP_SIZE[0]
        # local_pose3d_mesh3d_gt_deproj = cam_deprojection(factor * pose_mesh_2d[0][:,0:-1],
        #                                             cam_proj_mat,
        #                                             z=z)

        # numpy array of size 21 x 64 x 64. Elements range from 0.0 to 1.0
        target_heatmaps = self.heatmap_generator(pose2d) 

        #show
        # for k in range(0,21,6):
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(131)
        #     ax2 = fig.add_subplot(132)
        #     ax3 = fig.add_subplot(133)
        #     print('subpixel:',pose2d[k])
        #     print('pixel:',np.argmax(target_heatmaps[k]) - np.argmax(target_heatmaps[k]) // 64 * 64, np.argmax(target_heatmaps[k]) // 64)
        #     ax1.imshow(cv2.cvtColor(np.transpose((img-img.min()).numpy(),(1,2,0)), cv2.COLOR_RGB2BGR))
        #     ax3.imshow(ori_img)
        #     plot_hand(ax1, 4*pose2d[:,0:2], order='uv')
        #     plot_hand(ax2, pose2d[:,0:2], order='uv')
        #     ax2.imshow(target_heatmaps[k])
        #     plt.title('HandGraph: {} Joint id: {} Vis: {}'.format(idx, k, pose2d[k,2]==1))
        #     plt.show()

        # numpy array of size 256 x 256. Elements are either 255 or 0
        
        return img, target_heatmaps, pose2d
        # return img, img_mask, \
        #         target_heatmaps, \
        #         pose2d, \
        #         local_pose3d_mesh3d_gt_deproj[0:local_pose3d_gt.shape[0], :], \
        #         mesh_2d, \
        #         local_pose3d_mesh3d_gt_deproj[local_pose3d_gt.shape[0]:, :], \
        #         mesh_tri_idx, \
        #         cam_proj_mat, \
        #         img_path
        
    def visualize_data(self):
        for i in range(self.__len__()):
            # img type: float32; img_mask type uint8
            img, img_mask, target_heatmaps, pose2d, local_pose3d_gt, mesh_2d, local_mesh_pts_gt, mesh_tri_idx, cam_proj_mat = self.__getitem__(i)
            img = cv2.cvtColor(np.transpose(img.numpy(),(1,2,0)), cv2.COLOR_RGB2BGR)

            img_wo_bkg = cv2.bitwise_and(img, img, mask=img_mask)
            # for training data, you can use the hand mask to blend hand image with random background images
            
            im_height, im_width = img.shape[:2]
            fig = plt.figure(0)
            fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

            # 1. plot raw image
            ax1 = fig.add_subplot(2,3,1)
            ax1.imshow(img)
            ax1.set_title("raw image")

            # 2. plot image without background
            ax2 = fig.add_subplot(2,3,2)
            ax2.imshow(img_wo_bkg)
            ax2.set_title("image without background")

            # 3. plot 2D joints
            ax3 = fig.add_subplot(2,3,3)
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
            factor = self.config.MODEL.IMAGE_SIZE[0] / self.config.MODEL.HEATMAP_SIZE[0]
            skeleton_overlay = draw_2d_skeleton(img_mask, factor * pose2d[:,0:2])
            ax3.imshow(skeleton_overlay)
            ax3.set_title("image with GT 2D joints")

            # 4. plot 3D joints
            ax4 = fig.add_subplot(234, projection='3d')
            draw_3d_skeleton_on_ax(local_pose3d_gt, ax4)
            ax4.set_title("GT 3D joints")

            # 5. plot 3D mesh
            ax5 =fig.add_subplot(235, projection='3d')
            ax5.plot_trisurf(local_mesh_pts_gt[:, 0], local_mesh_pts_gt[:, 1], local_mesh_pts_gt[:, 2],
                            triangles=mesh_tri_idx, color='grey', alpha=0.8)
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.set_zlabel('Z')
            ax5.view_init(elev=-85, azim=-75)
            ax5.set_title("GT 3D mesh")

            # 6. plot 2D mesh points
            ax6 = fig.add_subplot(2,3,6)
            ax6.imshow(img)
            ax6.scatter(factor * mesh_2d[:, 0], factor * mesh_2d[:, 1], s=15, color='green', alpha=0.8)
            ax6.set_title("image with GT mesh 2D projection points")

            fig1 = plt.figure(1)
            for k in range(0,21,5):
                resized_hm = cv2.resize(cv2.cvtColor((255*target_heatmaps[k]).astype(np.uint8),cv2.COLOR_GRAY2BGR), tuple(self.config.MODEL.IMAGE_SIZE))
                mix = cv2.addWeighted(img_mask, 0.3, resized_hm, 0.7, 0)
                plt.title(self.joint_indices[k])
                plt.imshow(mix)
                plt.show()