import os
import pickle
import shutil
import time
import fnmatch
import logging
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from utils.misc import update_after_resize
from .standard_legends import idx_MHP
from .FHA_utils import *
from .HandGraph_utils.vis import *

logger = logging.getLogger(__name__)

def readAnnotation3D(file):
	f = open(file, "r")
	an = []
	for l in f:
		l = l.split()
		an.append((float(l[1]),float(l[2]), float(l[3])))

	return np.array(an, dtype=float)

class MHPMultiViewDataset(Dataset):
    """
    Split: 4:1
    """
    def __init__(self, config, set_name, heatmap_generator=None, transform=None):
        self.name = 'MHP'
        self.orig_img_size = [640,480]
        self.transform = transform
        self.heatmap_generator = heatmap_generator
        self.data_dir = os.path.join(config.DATA_DIR, 'MHP') # ../data/MHP
        self.anno_dir = 'annotations'
        # rearrange the order of the annotations of 21 joints
        self.reorder_idx = idx_MHP

        # load samples onto 
        self.rvec_dict, self.tvec_dict, self.data_list, self.pose3d_dict = {}, {}, [], {}
        if set_name in ['train', 'training']:
            subdir_range = range(1,17)
            # self.data_list += [os.path.join('augmented_samples','data_{}'.format(i)) for i in range(1,10)] \
            #     + [os.path.join('augmented_samples','data_11'), os.path.join('augmented_samples','data_14'), os.path.join('augmented_samples','data_15'), os.path.join('augmented_samples','data_16')]
            #self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(1,3)]
        elif set_name in ['eval', 'valid', 'val', 'evaluation', 'validation']:
            subdir_range = range(17,22)
            # self.data_list += [os.path.join('augmented_samples','data_17'), os.path.join('augmented_samples','data_18'), os.path.join('augmented_samples','data_19'), os.path.join('augmented_samples','data_21')]
            #self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(3,5)]
        for i in subdir_range:
            data_subdir = 'data_{}'.format(i)
            if data_subdir not in self.rvec_dict:
                self.rvec_dict[data_subdir] = {}
            if data_subdir not in self.tvec_dict:
                self.tvec_dict[data_subdir] = {}
            if data_subdir not in self.pose3d_dict:
                self.pose3d_dict[data_subdir] = {}
            data_dir = os.path.join(self.data_dir, 'annotated_frames', data_subdir)
            self.data_list.append(data_dir)
            for cam_idx in range(1,5):
                self.rvec_dict[data_subdir][str(cam_idx)] = pickle.load(open(os.path.join(self.data_dir, 'calibrations', data_subdir, 'webcam_{}'.format(cam_idx), 'rvec.pkl'), "rb"), encoding='latin1')
                self.tvec_dict[data_subdir][str(cam_idx)] = pickle.load(open(os.path.join(self.data_dir, 'calibrations', data_subdir, 'webcam_{}'.format(cam_idx), 'tvec.pkl'), "rb"), encoding='latin1')
            for frame_idx in range(len(os.listdir(data_dir)) // 4):
                    # ex: ../MHP/annotattions/data_1/0_joints.txt
                pose3d_path = os.path.join(self.data_dir, self.anno_dir, data_subdir, str(frame_idx) + '_joints.txt')
                self.pose3d_dict[data_subdir][frame_idx] = readAnnotation3D(pose3d_path)[self.reorder_idx] # 21 x 3

        self.cur_data_idx = 0
        self.cur_datadir_len = len(os.listdir(os.path.join(self.data_list[self.cur_data_idx]))) // 4
        self.cur_frame_idx = 0

        Fx, Fy, Cx, Cy = 614.878, 615.479, 313.219, 231.288

        # self.intrinsic_matrix = update_after_resize(
        #     np.array([[Fx, 0, Cx],
        #             [0, Fy, Cy],
        #             [0,  0, 1 ]], dtype='float32'),  # dtype('float64')
        #             image_shape=(480, 640), new_image_shape=self.config.MODEL.HEATMAP_SIZE
        # )
        self.intrinsic_matrix = np.array([[Fx, 0, Cx],
                                            [0, Fy, Cy],
                                            [0,  0, 1 ]], dtype='float32')
                                    
        self.distortion_coeffs = 0 * np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])
        

        
        # calculate the number of samples
        Nsamples = 0
        for data in self.data_list:
            frame_list = os.listdir(data)
            Nsamples += len(frame_list) // 4
        
        self.l = Nsamples
        self.n_views = [1,2,3,4]

    def __len__(self):
        return self.l
        
    def update(self):
        self.cur_frame_idx += 1

        if self.cur_frame_idx >= self.cur_datadir_len:
            #print('next camera')
            self.cur_frame_idx = 0

            if self.cur_data_idx < len(self.data_list) - 1:
                self.cur_data_idx += 1   
            else:
                # reset the indices for the next epoch
                self.cur_data_idx = 0
                self.cur_frame_idx = 0
                print('Run out of samples') 
            
            self.cur_datadir_len = len(os.listdir(os.path.join(self.data_list[self.cur_data_idx]))) // 4


    def __getitem__(self, i):
        ori_img_lst = []
        img_lst = []
        pose2d_lst = []
        heatmap_lst = []
        proj_matrix_lst = []
        visibility_lst = []
        # ex: ../MHP/annotated_frames/data1/0_joints.txt
        subdir = os.path.basename(self.data_list[self.cur_data_idx]) #
        pose3d_gt = self.pose3d_dict[subdir][self.cur_frame_idx].astype('float32') # 21 x 3

        #print(pose3d_path)
        for cam_idx in self.n_views:#range(1,1 + self.n_views):
            img_path = os.path.join(self.data_list[self.cur_data_idx], '{}_webcam_{}.jpg'.format(self.cur_frame_idx, cam_idx))
            #print(img_path)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # 480(H) x 640(W) x 3
            ori_img_lst.append(img)
            # load 3D pose (world coord)
            dir_name, img_name = os.path.split(img_path) # ex: ../MHP/annotated_frames/data_1, 0_webcam_1.jpg
            
            # load extrinsic params. rvec: rotation vectors (Rodrigues); tvec: translation vectors
            webcam_id = img_name[-5]
            rvec = self.rvec_dict[subdir][webcam_id] 
            tvec = self.tvec_dict[subdir][webcam_id]
            rotation_matrix, _ = cv2.Rodrigues(rvec) # dtype: float 64
            rigidMatrix = np.concatenate((np.float32(rotation_matrix), np.float32(tvec).reshape((3,1))), axis=1)
            # rigidMatrix = np.concatenate((rigidMatrix, np.array([[0,0,0,1]], dtype=np.float32)))
            # KH = self.intrinsic_matrix.dot(rigidMatrix)

            #rot_mat_lst.append(rotation_matrix)
            #tvec_lst.append(tvec)

            # project 3D points (world coord) onto the image plane
            pose3d_cam = np.dot(rotation_matrix, np.transpose(pose3d_gt, (1,0))) + tvec.reshape((3,1)) # 3 x 21
            
            # project 3D points (world coord) onto the image plane
            zeros = np.array([0.,0.,0.])
            pose2d, _ = cv2.projectPoints(
                objectPoints=pose3d_cam,
                rvec=zeros, tvec=zeros,
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.distortion_coeffs) # 21 x 1 x 2
            pose2d = pose2d.squeeze() # 21 x 2

            # random occlusion 
            if True:
                import random
                random.seed(4*i+cam_idx)
                radius = 50 # 30 50 70
                center = pose2d[random.randint(0,20)].astype(int) # select a keypoint as the circle center
                img = cv2.circle(img,center=tuple(center.tolist()),radius=radius,color=(0,0,0),thickness=-1)
            
            visibility = np.ones((pose2d.shape[0],1))
            img_height, img_width = img.shape[0:2]
            for k in range(pose2d.shape[0]):
                if pose2d[k,0] < 0 or pose2d[k,1] < 0 or pose2d[k,0] >= img_width or pose2d[k,1] >= img_height or np.linalg.norm(pose2d[k] - center) <= radius:
                    visibility[k] = 0

            img, pose2d = self.transform(img, [pose2d])
            pose2d = pose2d[0]

            if self.heatmap_generator is not None:
                heatmap_lst.append(self.heatmap_generator(np.concatenate((pose2d, visibility), axis=1)))
        
            img_lst.append(img)
            proj_matrix_lst.append(rigidMatrix)
            pose2d_lst.append(pose2d)
            visibility_lst.append(visibility)

        self.update()

        #self._visualize_data(imgs, pose2d_gt, pose3d_gts)

        # pose2d_gt: 4 x 21 x 2
        # visibility: 4 x 21 x 1

        ret = {
            'data_idx': self.cur_data_idx,
            'orig_imgs': np.stack(ori_img_lst),
            'imgs': np.stack(img_lst),
            'pose2d': np.stack(pose2d_lst),
            'pose3d': pose3d_gt,
            'visibility': np.stack(visibility_lst),
            'extrinsic_matrices': np.stack(proj_matrix_lst),
            'intrinsic_matrix': self.intrinsic_matrix,
        }

        if len(heatmap_lst) > 0:
            ret['heatmaps'] = np.stack(heatmap_lst)

        return ret

        #return ori_imgs, imgs, pose2d_gt, pose3d_gt, visibility, proj_matrices, self.intrinsic_matrix

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {} ({} views: {})\n'.format(self.__len__(), len(self.n_views), str(self.n_views))
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        return fmt_str

# class MHPMultiViewDataset(Dataset):
#     """
#     Split: 4:1
#     """
#     def __init__(self, config, set_name, transform):
#         self.name = 'MHP'
#         self.config = config
#         self.transform = transform
#         self.data_dir = os.path.join(config.DATA_DIR, 'MHP') # MHP
#         self.anno_dir = 'annotations'

#         if set_name in ['train', 'training']:
#             self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(1,17)]
#             # self.data_list += [os.path.join('augmented_samples','data_{}'.format(i)) for i in range(1,10)] \
#             #     + [os.path.join('augmented_samples','data_11'), os.path.join('augmented_samples','data_14'), os.path.join('augmented_samples','data_15'), os.path.join('augmented_samples','data_16')]
#             #self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(1,3)]
#         elif set_name in ['eval', 'valid', 'val', 'evaluation', 'validation']:
#             self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(17,22)]
#             # self.data_list += [os.path.join('augmented_samples','data_17'), os.path.join('augmented_samples','data_18'), os.path.join('augmented_samples','data_19'), os.path.join('augmented_samples','data_21')]
#             #self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(3,5)]
#         self.cur_data_idx = 0
#         self.cur_datadir_len = len(os.listdir(os.path.join(self.data_dir, self.data_list[self.cur_data_idx]))) // 4
#         self.cur_frame_idx = 0

#         Fx, Fy, Cx, Cy = 614.878, 615.479, 313.219, 231.288

#         self.intrinsic_matrix = np.array([[Fx, 0, Cx],
#                                           [0, Fy, Cy],
#                                           [0,  0, 1 ]], dtype='float32') # dtype('float64')

#         self.distortion_coeffs = 0 * np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])
        
#         # rearrange the order of the annotations of 21 joints
#         self.reorder_idx = idx_MHP
        
#         # calculate the number of samples
#         Nsamples = 0
#         for data in self.data_list:
#             frame_list = os.listdir(os.path.join(self.data_dir, data))
#             Nsamples += len(frame_list) // 4
        
#         self.l = Nsamples

#     def __len__(self):
#         return self.l
        
#     def update(self):
#         self.cur_frame_idx += 1

#         if self.cur_frame_idx >= self.cur_datadir_len:
#             #print('next camera')
#             self.cur_frame_idx = 0

#             if self.cur_data_idx < len(self.data_list) - 1:
#                 self.cur_data_idx += 1   
#             else:
#                 # reset the indices for the next epoch
#                 self.cur_data_idx = 0
#                 self.cur_frame_idx = 0
#                 print('Run out of samples') 
            
#             self.cur_datadir_len = len(os.listdir(os.path.join(self.data_dir, self.data_list[self.cur_data_idx]))) // 4


#     def __getitem__(self, i):
#         ori_img_lst = []
#         img_lst = []
#         pose2d_lst = []
#         extrinsic_lst = []
#         visibility_lst = []
#         # ex: ../MHP/annotated_frames/data1/0_joints.txt
#         pose3d_path = os.path.join(self.data_dir, self.anno_dir, os.path.basename(self.data_list[self.cur_data_idx]), str(self.cur_frame_idx) + '_joints.txt')
#         pose3d_gt = readAnnotation3D(pose3d_path)[self.reorder_idx] # 21 x 3
#         #print(pose3d_path)
#         for cam_idx in range(1,5):
#             img_path = os.path.join(self.data_dir, self.data_list[self.cur_data_idx], '{}_webcam_{}.jpg'.format(self.cur_frame_idx, cam_idx))
#             #print(img_path)
#             img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # 480(H) x 640(W) x 3
#             ori_img_lst.append(img)
#             # load 3D pose (world coord)
#             dir_name, img_name = os.path.split(img_path) # ex: ../MHP/annotated_frames/data_1, 0_webcam_1.jpg
            
#             # load extrinsic params. rvec: rotation vectors (Rodrigues); tvec: translation vectors
#             dir_id, webcam_id = dir_name.split('_')[-1], img_name[-5]
#             rvec = pickle.load(open(os.path.join(self.data_dir, 'calibrations', 'data_{}'.format(dir_id), 'webcam_{}'.format(webcam_id), 'rvec.pkl'), "rb"), encoding='latin1')
#             tvec = pickle.load(open(os.path.join(self.data_dir, 'calibrations', 'data_{}'.format(dir_id), 'webcam_{}'.format(webcam_id), 'tvec.pkl'), "rb"), encoding='latin1')
#             rotation_matrix, _ = cv2.Rodrigues(rvec)

#             #rot_mat_lst.append(rotation_matrix)
#             #tvec_lst.append(tvec)
 
#             # project 3D points (world coord) onto the image plane
#             pose3d_cam = np.dot(rotation_matrix, np.transpose(pose3d_gt, (1,0))) + tvec.reshape((3,1)) # 3 x 21
#             rigidMatrix = np.concatenate((rotation_matrix, tvec.reshape((3,1))), axis=1)
#             extrinsic_lst.append(rigidMatrix)

#             # project 3D points (world coord) onto the image plane
#             zeros = np.array([0.,0.,0.])
#             pose2d, _ = cv2.projectPoints(
#                 objectPoints=pose3d_cam,
#                 rvec=zeros, tvec=zeros,
#                 cameraMatrix=self.intrinsic_matrix,
#                 distCoeffs=self.distortion_coeffs) # 21 x 1 x 2

#             visibility = np.ones((pose2d.shape[0],1))
#             img_height, img_width = img.shape[0:2]
#             for k in range(pose2d.shape[0]):
#                 if pose2d[k,0,0] < 0 or pose2d[k,0,1] < 0 or pose2d[k,0,0] >= img_width or pose2d[k,0,1] >= img_height:
#                     visibility[k] = 0
#             visibility_lst.append(visibility)

#             img, _ = self.transform(img, [np.ones((21,2))])

#             img_lst.append(img)
#             pose2d_lst.append(np.squeeze(pose2d))

#         self.update()

#         ori_imgs = np.stack(ori_img_lst)
#         imgs = np.stack(img_lst)
#         pose2d_gt = np.stack(pose2d_lst)
#         extrinsics = np.stack(extrinsic_lst)
#         visibility = np.stack(visibility_lst)
#         #self._visualize_data(imgs, pose2d_gt, pose3d_gts)

#         return ori_imgs, imgs, pose2d_gt, pose3d_gt, visibility, extrinsics, self.intrinsic_matrix #, np.stack(rot_mat_lst), np.stack(tvec_lst)

#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of samples: {}\n'.format(self.__len__())
#         fmt_str += '    Root Location: {}\n'.format(self.data_dir)
#         return fmt_str
