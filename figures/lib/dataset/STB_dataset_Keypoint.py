import os
import shutil
import time

from .STB_dataset import STBDataset
from PIL import Image

import cv2
import numpy as np
import pickle
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.frei_utils.fh_utils import *

class STBDataset_Keypoint(STBDataset):
    def __init__(self, config, data_dir, heatmap_generator, transforms=None):
        super().__init__(config.DATASET.ROOT,
                data_dir,
                config.DATASET.DATA_FORMAT)
        
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
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        return fmt_str
    
    def __getitem__(self, idx):
        img, cam_params, pose_gts, bboxes, pose_roots, pose_scales, index = super().__getitem__(idx)
        # read the 2D keypoints annotation of this image. Note: "self.anno_all[idx]['uv_vis']"" is a numpy array of size 42x3.
        # The first colum of 2D kp annotation matrix represents the x-axis (horizontal and positive rightwards) value of a key point;
        # the second column represents the y-axis (vertical and positive upwards) value of a key point;
        # the third column consists boolean values donoting the visibility of key points (1 for visible points and 0 otherwise)

        if self.transforms: # for training and validation
            img, joints = self.transforms( # joints is a list, img size: 3 x H x W
                img, [pose_gts]
            )
            joints = joints[0]
            print(joints.shape)
            target_heatmaps = self.heatmap_generator(joints) # numpy array of size 21 x 64 x 64

            # show
            for i in range(21):
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                print(joints[i])
                print(np.argmax(target_heatmaps[i]) - np.argmax(target_heatmaps[i]) // 64 * 64, np.argmax(target_heatmaps[i]) // 64)
                ax1.imshow(np.transpose(img.numpy(),(1,2,0)))
                plot_hand(ax1, 4*joints[:,0:2], order='uv')
                plot_hand(ax2, 4*joints[:,0:2], order='uv')
                ax2.imshow(target_heatmaps[i])
                plt.show()

            return img, target_heatmaps, joints, img_path
        
        else: # for inference
            target_heatmaps = self.heatmap_generator(joints)
            # show
            for i in range(21):
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                print(joints[i])
                print(np.argmax(target_heatmaps[i]) - np.argmax(target_heatmaps[i]) // 64 * 64, np.argmax(target_heatmaps[i]) // 64)
                ax1.imshow(img)
                plot_hand(ax1, joints[:,0:2], order='uv')
                plot_hand(ax2, joints[:,0:2], order='uv')
                ax2.imshow(target_heatmaps[i])
                plt.show()

            img = img.astype(np.float32)
            return np.transpose(img,(2,0,1)), target_heatmaps, joints, img_path

