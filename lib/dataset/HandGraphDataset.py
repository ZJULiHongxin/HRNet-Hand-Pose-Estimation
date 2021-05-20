import os
import shutil
import time
import logging
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from .standard_legends import std_legend_lst, idx_Frei
from .HandGraph_utils.utils import *
from .HandGraph_utils.vis import *

logger = logging.getLogger(__name__)

class HandGraphDataset(Dataset):
    """ 《3D Hand Shape and Pose Estimation From a Single RGB Image》
    ① 该数据集没有显示划分训练集和验证集，而是在3D_labels文件夹下的val-camera.txt里制定了用于验证的
    照相机视角。可调用get_train_val_paths.py获取训练集和验证集的图片路径
    Args:
        root (string): Root directory where dataset is located to.
        set_name (string): Dataset name('training', 'evaluation').
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, set_name, data_format, transform=None,
                 target_transform=None):
        self.name = 'HandGraph'
        self.data_dir = os.path.join(root, self.name) # HandGraph/training 
        self.set_name = set_name
        self.transform = transform
        self.target_transform = target_transform

        camera_param_path = osp.join(self.data_dir,'3D_labels/camPosition.txt')
        global_pose3d_gt_path = osp.join(self.data_dir, '3D_labels/handGestures.txt')
        
        val_set_path = osp.join(self.data_dir, '3D_labels/val-camera.txt')

        self.image_dir = osp.join(self.data_dir, 'images')
        self.global_mesh_gt_dir = osp.join(self.data_dir, 'hand_3D_mesh')
        self.image_paths = self.get_train_val_im_paths(self.image_dir, val_set_path, set_name)
        self.all_camera_params, self.all_global_pose3d_gt = \
            self.init_pose3d_labels(camera_param_path, global_pose3d_gt_path)  # all_camera_params:  Nsamples x Ncams x 7; all_global_pose3d_gt: Nsamples x Njoints x 3


    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_path = self.image_paths[idx]

        # get 3D pose ground truth
        pose_id, camera_id = extract_pose_camera_id(osp.basename(img_path))
        cam_param = self.all_camera_params[pose_id][camera_id]  # (7, )

        global_pose3d_gt = self.all_global_pose3d_gt[pose_id]  # (21, 3)
        local_pose3d_gt = transform_global_to_cam(global_pose3d_gt, cam_param)  # (21, 3)

        # local_pose3d_gt, local_mesh_pts_gt, local_mesh_normal_gt, cam_param, mesh_tri_idx = \
        #     self.read_data(img_path, self.all_camera_params, self.all_global_pose3d_gt, self.global_mesh_gt_dir)
        
        img_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # 360 x 360 x 4 (RGBA)
        img, img_mask = img_rgba[:,:,0:3], img_rgba[:,:,3:]

        im_height, im_width = img.shape[:2]

        fl = cam_param[0]
        cam_proj_mat = np.array([[fl,  0.0, im_width / 2.],
                                 [0.0, fl,  im_height / 2.],
                                 [0.0, 0.0, 1.0]])    

        pose2d = cam_projection(local_pose3d_gt, cam_proj_mat) # (N, 2)
        #mesh_2d = cam_projection(local_mesh_pts_gt, cam_proj_mat)

        # append a column representing visibility
        visibility = np.ones((21,1), dtype=pose2d.dtype)
        # for k in range(21):
        #     if pose2d[k,0] >= im_width or pose2d[k,0] < 0 or pose2d[k,1] >= im_height or pose2d[k,1] < 0:
        #         visibility[k,0] = 0
        pose2d = np.concatenate((pose2d, visibility), axis=1)
        #mesh_2d = np.concatenate((mesh_2d, np.ones((mesh_2d.shape[0],1))), axis=1)
        
        if self.transform is not None:
            img, joints_list = self.transform(img,[pose2d])
            return img, joints_list[0], img_path

        if self.target_transform is not None:
            pose2d = self.target_transform(pose2d)
        
        return img, pose2d, img_path
        #return img, img_mask, pose2d, local_pose3d_gt, mesh_2d, local_mesh_pts_gt, mesh_tri_idx, cam_proj_mat, img_path
        

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def init_pose3d_labels(self, cam_param_path, pose3d_gt_path):
        all_camera_params = load_camera_param(cam_param_path)
        all_global_pose3d_gt = load_global_pose3d_gt(pose3d_gt_path)
        return all_camera_params, all_global_pose3d_gt

    def get_train_val_im_paths(self, image_dir, val_set_path, train_val_flag):
        """
        get training or validation image paths
        :param image_dir:
        :param val_set_path:
        :param train_val_flag:
        :return:
        """
        val_cameras = []
        with open(val_set_path) as reader:
            for line in reader:
                val_cameras.append(line.strip())
        val_cameras = set(val_cameras)

        lighting_folders = glob.glob(osp.join(image_dir, "l*"))

        image_paths = []
        for lighting_folder in lighting_folders:
            cam_folders = glob.glob(osp.join(lighting_folder, "cam*"))
            for cam_folder in cam_folders:
                cam_name = osp.basename(cam_folder)
                is_val = (cam_name in val_cameras)
                if ('val' in train_val_flag and is_val) or \
                        ('train' in train_val_flag and not is_val):
                    image_paths += glob.glob(osp.join(cam_folder, "*.png"))

        return image_paths

    def read_data(self, im_path, all_camera_params, all_global_pose3d_gt, global_mesh_gt_dir):
        """
        read the corresponding pose and mesh ground truth of the image sample, and camera parameters
        :param im_path:
        :param all_camera_params: (N_pose, N_cam, 7) focal_length, 3 translation val; 3 euler angles (degree)
        :param all_global_pose3d_gt: (N_pose, 21, 3)
        :param global_mesh_gt_dir:
        :return:
        """
        pose_id, camera_id = extract_pose_camera_id(osp.basename(im_path))

        cam_param = all_camera_params[pose_id][camera_id]  # (7, )

        # get ground truth of 3D hand pose
        global_pose3d_gt = all_global_pose3d_gt[pose_id]  # (21, 3)
        local_pose3d_gt = transform_global_to_cam(global_pose3d_gt, cam_param)  # (21, 3)

        # get ground truth of 3D hand mesh
        mesh_files = glob.glob(osp.join(global_mesh_gt_dir, "*.%04d.obj" % (pose_id + 1)))
        assert len(mesh_files) == 1, "Cannot find a unique mesh file for pose %04d" % (pose_id + 1)
        mesh_file = mesh_files[0]
        global_mesh_pts_gt, global_mesh_normal_gt, mesh_tri_idx = load_mesh_from_obj(mesh_file)
        # global_mesh_pts_gt: (N_vertex, 3), global_mesh_normal_gt: (N_tris, 3)
        # mesh_tri_idx: (N_tris, 3)

        local_mesh_pts_gt = transform_global_to_cam(global_mesh_pts_gt, cam_param)  # (N_vertex, 3)
        local_mesh_normal_gt = transform_global_to_cam(global_mesh_normal_gt, cam_param)

        return local_pose3d_gt, local_mesh_pts_gt, local_mesh_normal_gt, cam_param, mesh_tri_idx
    
    def visualize_data(self):
        for i in range(self.__len__()):
            img, pose2d, local_pose3d_gt, mesh_2d, local_mesh_pts_gt, mesh_tri_idx = self.__getitem__(i)
            img_rgb = img[:, :, :3]
            img_mask = img[:, :, 3:]
            img_wo_bkg = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask)
            # for training data, you can use the hand mask to blend hand image with random background images
            
            im_height, im_width = img_rgb.shape[:2]
            fig = plt.figure()
            fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

            # 1. plot raw image
            ax1 = fig.add_subplot(2,3,1)
            ax1.imshow(img_rgb)
            ax1.set_title("raw image")

            # 2. plot image without background
            ax2 = fig.add_subplot(2,3,2)
            ax2.imshow(img_wo_bkg)
            ax2.set_title("image without background")

            # 3. plot 2D joints
            ax3 = fig.add_subplot(2,3,3)
            skeleton_overlay = draw_2d_skeleton(img_rgb, pose2d)
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
            ax6.imshow(img_rgb)
            ax6.scatter(mesh_2d[:, 0], mesh_2d[:, 1], s=15, color='green', alpha=0.8)
            ax6.set_title("image with GT mesh 2D projection points")

            plt.show()
    
    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def evaluate(self, config, preds, scores, output_dir,
                 *args, **kwargs):
        '''
        Perform evaluation on RHD keypoint task
        :param config: config dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.data_dir)

        # preds is a list of: batchsize x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)
                # if self.with_center:

                kpts[int(file_name[-16:-4])].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'tags': kpt[:, 3],
                        'image': int(file_name[-16:-4]),
                        'area': area
                    }
                )

        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        if 'test' not in self.set_name:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder
            )
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def generate_videos(self):
        import random
        output_dir = os.path.join(self.data_dir, 'videos')

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        sample_num = self.__len__()
        video_num = 1
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_num = 100
        tol = 0.5
        video_count = 0
        for i in range(sample_num):
            if video_count == video_num:
                break
            # starting pose
            img_path = os.path.join(self.data_dir, 'rgb', self.images[i])
            mask_path = os.path.join(self.data_dir, 'mask', self.masks[i])
        
            imgk = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            imgk = cv2.cvtColor(imgk, cv2.COLOR_BGR2RGB)

            maskk = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
            Kk, mano, xyzk = self.db_data_anno[i]
            Kk, _, xyzk = [np.array(x) for x in [Kk, mano, xyzk]]

            self.maskout(imgk, maskk)

            # end pose
            end_idx = random.randint(0, sample_num)
            img_path = os.path.join(self.data_dir, 'rgb', self.images[end_idx])
            mask_path = os.path.join(self.data_dir, 'mask', self.masks[end_idx])
        
            imgN = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            maskN = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

            KN, mano, xyzN = self.db_data_anno[i]
            KN, _, xyzN = [np.array(x) for x in [KN, mano, xyzN]]
            
            self.maskout(imgN, maskN)

            video_frames = [imgk]

            count = 1
            flag = 1

            id_visitd = [i,end_idx]
            while True:
                temp_xyzk = xyzk - count /(frame_num-1) * (xyzk - xyzN)
                print(count)
                for j in range(sample_num):
                    if j in id_visitd:
                        continue
                    id_visitd.append(j)
                    Kk, mano, xyzk = self.db_data_anno[j]
                    _, _, xyzk = [np.array(x) for x in [Kk, mano, xyzk]]

                    dist = np.sum(np.linalg.norm(xyzk - temp_xyzk, axis=1))

                    if dist < tol:
                        img_path = os.path.join(self.data_dir, 'rgb', self.images[j])
                        mask_path = os.path.join(self.data_dir, 'mask', self.masks[j])
                        imgk = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        maskk = cv2.cvtColor(cv2.imread(mask_path, -1), cv2.COLOR_BGR2GRAY)

                        self.maskout(imgk, maskk)
                        # cv2.imshow('1',imgk)
                        # cv2.waitKey(0)
                        video_frames.append(imgk)
                        count += 1
                        break
                else:
                    flag = 0

                if flag == 0:
                    del video_frames
                    break
                if count == frame_num - 2:
                    break
            
            if flag:
                video_path = os.path.join(output_dir, 'VIDEO_{:06d}.avi'.format(video_count))
                output_movie = cv2.VideoWriter(video_path, fourcc, 5, (224,224))
                for frame in video_frames:
                    output_movie.write(frame)
                output_movie.write(imgN)
                print(video_path,'finished')
                output_movie.release()
                video_count += 1

    def maskout(self, img, mask):
        for ch in range(3):
            img[:,:,ch] = np.bitwise_and(img[:,:,ch], mask)
        
    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []
        num_joints = 17

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            for k in range(len(img_kpts)):
                kpt = key_points[k].reshape((num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'bbox': list([left_top[0], left_top[1], w, h])
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
            # info_str.append(coco_eval.stats[ind])

        return info_str


