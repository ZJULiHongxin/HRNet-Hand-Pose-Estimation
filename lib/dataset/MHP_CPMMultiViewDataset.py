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
from .transforms import Mytransforms
from utils.misc import update_after_resize
from .standard_legends import idx_MHP
from dataset.frei_utils.fh_utils import plot_hand

logger = logging.getLogger(__name__)

def readAnnotation3D(file):
	f = open(file, "r")
	an = []
	for l in f:
		l = l.split()
		an.append((float(l[1]),float(l[2]), float(l[3])))

	return np.array(an, dtype=float)

def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class MHP_CPMMultiViewDataset(Dataset):
    """
    Split: 4:1
    """
    def __init__(self, config, set_name, heatmap_generator=None, transform=None):
        self.name = 'MHP'
        self.hm_size = config.MODEL.HEATMAP_SIZE[0] # 64
        self.input_size = config.MODEL.IMAGE_SIZE[0] # 256
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
        
        self.transform = Mytransforms.Compose([Mytransforms.TestResized(256)])

        # calculate the number of samples
        Nsamples = 0
        for data in self.data_list:
            frame_list = os.listdir(data)
            Nsamples += len(frame_list) // 4
        
        self.l = Nsamples
        self.n_views = [1,2,3,4]
        self.sigma = config.DATASET.SIGMA
        self.exception = False

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
        centermap_lst = []
        proj_matrix_lst = []
        visibility_lst = []
        # ex: ../MHP/annotated_frames/data1/0_joints.txt
        subdir = os.path.basename(self.data_list[self.cur_data_idx]) #
        pose3d_gt = self.pose3d_dict[subdir][self.cur_frame_idx] # 21 x 3
        self.exception = False
        #print(pose3d_path)
        for cam_idx in self.n_views:#range(1,1 + self.n_views):
            img_path = os.path.join(self.data_list[self.cur_data_idx], '{}_webcam_{}.jpg'.format(self.cur_frame_idx, cam_idx))
            #print(img_path)
            orig_img = cv2.imread(img_path) # 480(H) x 640(W) x 3 -> 256 x 256
            img = cv2.resize(orig_img, (self.input_size, self.input_size))
            ori_img_lst.append(orig_img)
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
            pose2d[:,0] *= (self.input_size / self.orig_img_size[0]) # 640 -> 256
            pose2d[:,1] *= (self.input_size / self.orig_img_size[1]) # 480 -> 256

            h, w = img.shape[0:2]
            visibility = np.ones((pose2d.shape[0],1))
            for k in range(pose2d.shape[0]):
                if pose2d[k,0] < 0 or pose2d[k,1] < 0 or pose2d[k,0] >= w or pose2d[k,1] >= h:
                    visibility[k] = 0
            pose2d = np.concatenate((pose2d, visibility), axis=1)

            try:
                center_x = (pose2d[np.argwhere(pose2d[:,0] < w), 0].max() +
                        pose2d[np.argwhere(pose2d[:,0] > 0), 0].min()) / 2
            except:
                center_x = w / 2
                self.exception = True
            try:
                center_y = (pose2d[np.argwhere(pose2d[:,1] < h), 1].max() +
                            pose2d[np.argwhere(pose2d[:,1] > 0), 1].min()) / 2
            except:
                center_y = h / 2
                self.exception = True
            center = [center_x, center_y]

            try:
                scale = (pose2d[np.argwhere(pose2d[:,1] < h), 1].max() -
                        pose2d[np.argwhere(pose2d[:,1] > 0), 1].min() + 4) / h
            except:
                scale = 0.5

            # expand dataset. pose2d 21 x 3 [u,v,vis]
            img, pose2d, center = self.transform(img, pose2d, center, scale)

            factor = self.input_size / self.hm_size
            proj_matrix_lst.append(rigidMatrix)
            pose2d_lst.append(pose2d[:,0:2] / factor)
            visibility_lst.append(visibility)
            
            heatmap = np.zeros((len(pose2d) + 1, self.hm_size, self.hm_size), dtype=np.float32)
            for i in range(len(pose2d)):
                # resize from 368 to 46
                x = int(pose2d[i][0]) * 1.0 / factor
                y = int(pose2d[i][1]) * 1.0 / factor
                heat_map = guassian_kernel(size_h=self.hm_size, size_w=self.hm_size, center_x=x, center_y=y, sigma=self.sigma)
                heat_map[heat_map > 1] = 1
                heat_map[heat_map < 0.0099] = 0
                heatmap[i + 1, :, :] = heat_map

            heatmap[0, :, :] = 1.0 - np.max(heatmap[1:, :, :], axis=0)  # for background

            # import matplotlib.pyplot as plt
        
            # for k in range(0,21,1):
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(121)
            #     ax2 = fig.add_subplot(122)
            #     print('subpixel:',pose2d[k])
            #     ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #     plot_hand(ax1, pose2d[:,0:2], order='uv')
            #     ax2.imshow(heatmap[k])
            #     plot_hand(ax2, pose2d[:,0:2] / factor, order='uv')
            #     plt.title('{} Joint id: {} Vis: {}'.format(img_path, k, pose2d[k,2]==1))
            #     plt.show()
            
            centermap = np.zeros((h, w, 1), dtype=np.float32)
            center_map = guassian_kernel(size_h=h, size_w=w, center_x=center[0], center_y=center[1], sigma=3)
            center_map[center_map > 1] = 1
            center_map[center_map < 0.0099] = 0
            centermap[:, :, 0] = center_map

            img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                        [256.0, 256.0, 256.0])

            centermap = Mytransforms.to_tensor(centermap)
            img_lst.append(img)
            heatmap_lst.append(heatmap)
            centermap_lst.append(centermap)
        self.update()

        #self._visualize_data(imgs, pose2d_gt, pose3d_gts)

        # pose2d_gt: 4 x 21 x 2
        # visibility: 4 x 21 x 1
        ret = {
            'data_idx': self.cur_data_idx,
            'orig_imgs': np.stack(ori_img_lst),
            'imgs': np.stack(img_lst),
            'heatmaps': np.stack(heatmap_lst),
            'centermaps': torch.stack(centermap_lst),
            'pose2d': np.stack(pose2d_lst),
            'pose3d': pose3d_gt,
            'visibility': np.stack(visibility_lst),
            'extrinsic_matrices': np.stack(proj_matrix_lst),
            'intrinsic_matrix': self.intrinsic_matrix,
        }

        return ret

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {} ({} views: {})\n'.format(self.__len__(), len(self.n_views), str(self.n_views))
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        return fmt_str
