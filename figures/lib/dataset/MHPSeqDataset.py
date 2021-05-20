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

from .standard_legends import std_legend_lst, idx_MHP
from dataset.frei_utils.fh_utils import load_db_annotation, projectPoints, db_size, plot_hand

from .FHA_utils import *
from .HandGraph_utils.vis import *

logger = logging.getLogger(__name__)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def recursive_glob(rootdir='.', pattern='*'):
	matches = []
	for root, dirnames, filenames in os.walk(rootdir):
	  for filename in fnmatch.filter(filenames, pattern):
		  matches.append(os.path.join(root, filename))

	return matches

def readAnnotation3D(file):
	f = open(file, "r")
	an = []
	for l in f:
		l = l.split()
		an.append((float(l[1]),float(l[2]), float(l[3])))

	return np.array(an, dtype=float)

class MHPSeqDataset(Dataset):
    """
    Split: 4:1
    """
    def __init__(self, config, set_name, data_format, transform=None,
                 target_transform=None):
        self.name = 'MHP'

        self.data_dir = config.DATASET.ROOT # MHP
        self.anno_dir = 'annotations'

        if set_name in ['train', 'training']:
            self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(1,17)]
            # self.data_list += [os.path.join('augmented_samples','data_{}'.format(i)) for i in range(1,10)] \
            #     + [os.path.join('augmented_samples','data_11'), os.path.join('augmented_samples','data_14'), os.path.join('augmented_samples','data_15'), os.path.join('augmented_samples','data_16')]
            #self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(1,3)]
        elif set_name in ['eval', 'valid', 'val', 'evaluation', 'validation']:
            self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(17,22)]
            # self.data_list += [os.path.join('augmented_samples','data_17'), os.path.join('augmented_samples','data_18'), os.path.join('augmented_samples','data_19'), os.path.join('augmented_samples','data_21')]
            #self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(3,5)]
        self.cur_data_idx = 0
        self.cur_datadir_len = len(os.listdir(os.path.join(self.data_dir, self.data_list[self.cur_data_idx]))) // 4
        self.cur_cam_idx = 1
        self.cur_frame_idx = 0
        
        self.sample_length = config.DATASET.N_FRAMES
        self.sample_stride = config.DATASET.SAMPLE_STRIDE

        self.transform = transform

        Fx, Fy, Cx, Cy = 614.878, 615.479, 313.219, 231.288

        self.intrinsic_matrix = np.array([[Fx, 0, Cx],
                                          [0, Fy, Cy],
                                          [0,  0, 1 ]], dtype='float32') # dtype('float64')

        self.distortion_coeffs = 0 * np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])
        
        # rearrange the order of the annotations of 21 joints
        self.reorder_idx = idx_MHP
        
        # calculate the number of samples
        Nsamples = 0
        for data in self.data_list:
            frame_list = os.listdir(os.path.join(self.data_dir, data))
            N = 4 * ((len(frame_list) // 4 - self.sample_length) // self.sample_stride + 1)
            print(os.path.join(self.data_dir, data), '->', N)
            Nsamples += N
        
        self.i = 0 # counter
        self.l = Nsamples

        # reusable
        self.entire_seq = True

    def __len__(self):
        return self.l
        

    def update(self):
        self.cur_frame_idx += self.sample_stride
        self.entire_seq = False

        if self.cur_frame_idx + self.sample_length > self.cur_datadir_len:
            #print('next camera')
            self.cur_frame_idx = 0
            self.entire_seq = True
            if self.cur_cam_idx < 4:
                self.cur_cam_idx += 1
            else:
                self.cur_cam_idx = 1
                #print('next data dir')
                if self.cur_data_idx < len(self.data_list) - 1:
                    self.cur_data_idx += 1   
                else:
                    # reset the indices for the next epoch
                    self.cur_data_idx = 0
                    self.cur_frame_idx = 0
                    self.cur_cam_idx = 1
                    print('Run out of samples') 
                
                self.cur_datadir_len = len(os.listdir(os.path.join(self.data_dir, self.data_list[self.cur_data_idx]))) // 4


    def __getitem__(self, i):
        img_lst = []
        pose2d_lst = []
        pose3d_lst = []
        # rot_mat_lst = []
        # tvec_lst = []

        # The same frames will not be loaded twice
        start_idx = self.cur_frame_idx if self.entire_seq \
            else self.cur_frame_idx + max(self.sample_length - self.sample_stride, 0)
        end_idx = start_idx + self.sample_length if self.entire_seq \
            else start_idx + min(self.sample_length, self.sample_stride)
        self.i += 1
        #print(self.i,'/',self.l, 'cur_fidx',self.cur_frame_idx,'start', start_idx, 'end', end_idx)

        for idx in range(start_idx, end_idx):
            img_path = os.path.join(self.data_dir, self.data_list[self.cur_data_idx], '{}_webcam_{}.jpg'.format(idx, self.cur_cam_idx))
            #print(img_path)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
            # load 3D pose (world coord)
            dir_name, img_name = os.path.split(img_path) # ex: ../MHP/annotated_frames/data_1, 0_webcam_1.jpg
            # ex: ../MHP/annotated_frames/data1/0_joints.txt
            pose3d_path = os.path.join(self.data_dir, self.anno_dir, os.path.basename(dir_name), str(idx) + '_joints.txt')
            pose3d = readAnnotation3D(pose3d_path)[self.reorder_idx] # 21 x 3

            # load extrinsic params. rvec: rotation vectors (Rodrigues); tvec: translation vectors
            dir_id, webcam_id = dir_name.split('_')[-1], img_name[-5]
            rvec = pickle.load(open(os.path.join(self.data_dir, 'calibrations', 'data_{}'.format(dir_id), 'webcam_{}'.format(webcam_id), 'rvec.pkl'), "rb"), encoding='latin1')
            tvec = pickle.load(open(os.path.join(self.data_dir, 'calibrations', 'data_{}'.format(dir_id), 'webcam_{}'.format(webcam_id), 'tvec.pkl'), "rb"), encoding='latin1')
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            #rot_mat_lst.append(rotation_matrix)
            #tvec_lst.append(tvec)
 
            # project 3D points (world coord) onto the image plane
            pose3d_cam = np.dot(rotation_matrix, np.transpose(pose3d, (1,0))) + tvec.reshape((3,1)) # 3 x 21

            # project 3D points (world coord) onto the image plane
            zeros = np.array([0.,0.,0.])
            pose2d, _ = cv2.projectPoints(
                objectPoints=pose3d_cam,
                rvec=zeros, tvec=zeros,
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.distortion_coeffs) # 21 x 1 x 2

            pose2d_vis = np.concatenate((np.squeeze(pose2d), np.ones((21,1))), axis=1) # 21 x 3

            img, pose2d = self.transform(img, [pose2d_vis])

            img_lst.append(img)
            pose2d_lst.append(4*pose2d[0])
            pose3d_lst.append(pose3d_cam)

        self.update()

        imgs = np.stack(img_lst)
        pose2d_gts = np.stack(pose2d_lst)
        pose3d_gts = np.transpose(np.stack(pose3d_lst), (0,2,1))
        #self._visualize_data(imgs, pose2d_gts, pose3d_gts)

        return imgs, pose2d_gts, pose3d_gts#, np.stack(rot_mat_lst), np.stack(tvec_lst)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Sample length: {}\n'.format(self.sample_length)
        fmt_str += '    Sample stride: {}\n'.format(self.sample_stride)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
       #fmt_str = fmt_str + 'Data directories: ' + str(self.data_list)
        return fmt_str
    
    def visualize_data(self):
        for i in range(self.__len__()):
            # img (torch.tensor): N_frames x 3 x H x W
            # pose2d_gts: N_frames x 21 x 3 [u,v,vis]
            # pose3d_gts: N_frames x 21 x 3 (cam coord)
            imgs, pose2d_gts, pose3d_gts = self.__getitem__(i)
            self._visualize_data(imgs, pose2d_gts, pose3d_gts)

    def _visualize_data(self, imgs, pose2d_gts, pose3d_gts):
        # img (torch.tensor): N_frames x 3 x H x W
        # target_heatmaps: N_frames x 21 x 64(H) x 64(W)
        # pose2d_gts: N_frames x 21 x 3 [u,v,vis]
        # pose3d_gts: N_frames x 21 x 3 (cam coord)

        im_height, im_width = imgs.shape[:2]

        for frame_id in range(imgs.shape[0]):
            img = cv2.cvtColor(np.transpose(imgs[frame_id],(1,2,0)), cv2.COLOR_RGB2BGR)
            pose2d = pose2d_gts[frame_id]
            pose3d = pose3d_gts[frame_id]
            #pose3d = pose3d - pose3d[0]
            print(pose3d)

            fig = plt.figure(frame_id)
            # fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

            # 1. plot raw image
            ax = fig.add_subplot(1,2,1)  
            ax.imshow(img)
            
            for i in range(21):
                if pose2d[i,2] == 0:
                    plt.scatter(pose2d[i,0],pose2d[i,1])
            visualize_joints_2d(ax, pose2d, joint_idxs=False)

            # 3. plot 3D joints
            ax = fig.add_subplot(122, projection='3d')
            draw_3d_skeleton_on_ax(pose3d, ax)
            ax.set_title("GT 3D joints")

            plt.show()
