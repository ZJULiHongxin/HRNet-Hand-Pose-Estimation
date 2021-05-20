import os
import shutil
import time
import fnmatch
import pickle
import re
import logging
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from .standard_legends import std_legend_lst, idx_MHP
from dataset.frei_utils.fh_utils import load_db_annotation, projectPoints, db_size, plot_hand

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

class MHPDataset(Dataset):
    """
    Split: 4:1
    """
    def __init__(self, root, set_name, data_format=None, transform=None,
                 target_transform=None):
        self.name = 'MHP'
        self.orig_img_size = [640, 480]
        self.data_dir = os.path.join(root, self.name) # FreiHAND

        self.image_paths = recursive_glob(root, "*_webcam_[0-9]*")
        self.image_paths = natural_sort(self.image_paths)
        self.set_name = set_name
        self.split = 0.8 # According to the dataset paper, the 20% for the test split and the remaining 80% for the training split.

        if set_name in ['train', 'training']:
            self.start_idx = 0
            self.end_idx = int(len(self.image_paths) * self.split)
        elif set_name in ['eval', 'valid', 'val', 'evaluation', 'validation']:
            self.start_idx = int(len(self.image_paths) * self.split)
            self.end_idx = len(self.image_paths)

        self.transform = transform
        self.target_transform = target_transform

        Fx, Fy, Cx, Cy = 614.878, 615.479, 313.219, 231.288

        self.intrinsic_matrix = np.array([[Fx, 0, Cx],
                                          [0, Fy, Cy],
                                          [0,  0, 1 ]])

        self.distortion_coeffs = np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])
        
        # rearrange the order of the annotations of 21 joints
        self.reorder_idx = idx_MHP
        

    def __len__(self):
        return self.end_idx - self.start_idx


    def __getitem__(self, idx):
        img_path = self.image_paths[self.start_idx + idx]
        img = cv2.imread(img_path)

        # load 3D pose (world coord)
        dir_name, img_name = os.path.split(img_path) # ex: ../MHP/annotated_frames/data_1, 0_webcam_1.jpg
        dir_id = dir_name.split('_')[-1]
        img_idx, _, webcam_id = img_name[0:-4].split('_')

        # ex: ../MHP/annotated_frames/data1/0_joints.txt
        pose3d_path = os.path.join(self.data_dir, 'annotations', os.path.basename(dir_name), img_idx + '_joints.txt')
        pose3d = readAnnotation3D(pose3d_path)[self.reorder_idx]

        # load extrinsic params
        rvec = pickle.load(
            open(
                os.path.join(
                    self.data_dir, 'calibrations', 'data_{}'.format(dir_id), 'webcam_{}'.format(webcam_id), 'rvec.pkl'), "rb"), encoding='latin1')
        tvec = pickle.load(open(os.path.join(self.data_dir, 'calibrations', 'data_{}'.format(dir_id), 'webcam_{}'.format(webcam_id), 'tvec.pkl'), "rb"), encoding='latin1')

        pose2d, _ = cv2.projectPoints(pose3d, rvec, tvec, self.intrinsic_matrix, self.distortion_coeffs) # 21 x 1 x 2
        pose2d = pose2d.squeeze()

        visibility = np.ones((pose2d.shape[0],1))
        img_height, img_width = img.shape[0:2]
        for k in range(pose2d.shape[0]):
            if pose2d[k,0] < 0 or pose2d[k,1] < 0 or pose2d[k,0] >= img_width or pose2d[k,1] >= img_height:
                visibility[k] = 0

        if self.transform is not None:
            img, pose2d = self.transform(img,[pose2d])
            pose2d = pose2d[0]
        
        if self.target_transform is not None:
            pose2d = self.target_transform(pose2d)
        
        ret = {
            'imgs': img,
            'pose2d': pose2d,
            'visibility': visibility
        }
        
        return ret

    def __repr__(self):
        fmt_str = '{} Dataset '.format(self.set_name.title()) + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



