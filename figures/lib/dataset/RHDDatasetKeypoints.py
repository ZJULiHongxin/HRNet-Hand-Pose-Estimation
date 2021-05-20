import os
import shutil
import time

from .RHDDataset import RHDDataset
from PIL import Image

import cv2
import numpy as np
import pickle
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import kornia
from kornia.geometry import spatial_soft_argmax2d
from kornia.geometry.subpix import dsnt

class RHDDataset_Keypoint(RHDDataset):
    def __init__(self, config, data_dir, heatmap_generator, transforms=None):
        r"""Read RGB images and their ground truth heat maps
        params:
        @ data_dir: the name of the training or validation data directory
        @ mode: indicates whether the this is a training set ot a test set
        @ transform: data augmentation is implemented if transformers are given
        """
        super().__init__(config.DATASET.ROOT,
                        data_dir,
                        config.DATASET.DATA_FORMAT)

        assert config.DATASET.NUM_JOINTS == 21, 'Number of joint for RHD is 21!'
        self.num_joints = config.DATASET.NUM_JOINTS
        self.image_size = np.array(config.MODEL.IMAGE_SIZE)
        self.heatmap_generator = heatmap_generator
        self.transforms = transforms
        self.scale_factor = config.DATASET.SCALE_FACTOR
        self.rotation_factor = config.DATASET.ROT_FACTOR
        self.flip = config.DATASET.FLIP

        self.config = config
        with open(self.anno2d_path, 'rb') as f:
            self.anno_all = pickle.load(f)  # type: dict
        
        #self.__repr__()

    def __len__(self):
        return len(self.images)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
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
        img, joints, img_path = super().__getitem__(idx)
        # read the 2D keypoints annotation of this image. Note: "self.anno_all[idx]['uv_vis']"" is a numpy array of size 42x3.
        # The first colum of 2D kp annotation matrix represents the x-axis (horizontal and positive rightwards) value of a key point;
        # the second column represents the y-axis (vertical and positive upwards) value of a key point;
        # the third column consists boolean values donoting the visibility of key points (1 for visible points and 0 otherwise)
        if self.transforms: # for training and validation
            img, joints = self.transforms( # joints is a list, img size: 3 x H x W
                img, [joints]
            )
        
            target_heatmaps = self.heatmap_generator(joints[0])

            return img, target_heatmaps, joints[0], img_path
        
        else: # for inference
            target_heatmaps = self.heatmap_generator(joints)
            img = img.astype(np.float32)
            return np.transpose(img,(2,0,1)), target_heatmaps, joints, img_path

