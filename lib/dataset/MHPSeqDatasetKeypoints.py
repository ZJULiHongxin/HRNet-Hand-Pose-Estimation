import os
import shutil
import time
import random
from .MHPSeqDataset import MHPSeqDataset
from PIL import Image

import cv2
import numpy as np
import pickle
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.frei_utils.fh_utils import *

class MHPSeqDataset_keypoint(MHPSeqDataset):
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

        self.config = config
            
        self.num_joints = config.DATASET.NUM_JOINTS
        self.image_size = np.array(config.MODEL.IMAGE_SIZE)
        self.heatmap_generator = heatmap_generator
        self.transforms = transforms

    def __len__(self):
        return super().__len__()
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)

        return fmt_str
    
    def __getitem__(self, idx):
        # imgs: N_frams x H x W x 3
        # pose2d: N_frams x21 x 3 [u,v,vis]
        # pose3d: N_frams x21 x 3 [Xw,Yw,Zw]]
        imgs, pose2d, pose3d = super().__getitem__(idx)

        if self.transforms: # for training and validation
            img, pose2d = self.transforms( # joints is a list, img size: 3 x H x W
                img, [pose2d]
            )
            pose2d = pose2d[0]

            target_heatmaps = self.heatmap_generator(pose2d) # numpy array of size 21 x 64 x 64

            # show
            # fig = plt.figure()
            # for i in range(0,21):
            #     plt.imshow(np.transpose(img.numpy(),(1,2,0)))
            #     plt.scatter(4*pose2d[i][0], 4*pose2d[i][1])
            #     plt.show()

            # for i in range(0,21,6):
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(121)
            #     ax2 = fig.add_subplot(122)
            #     print(pose2d[i])
            #     print(np.argmax(target_heatmaps[i]) - np.argmax(target_heatmaps[i]) // 64 * 64, np.argmax(target_heatmaps[i]) // 64)
            #     ax1.imshow(np.transpose(img.numpy(),(1,2,0)))
            #     plot_hand(ax1, 4*pose2d[:,0:2], order='uv')
            #     plot_hand(ax2, 4*pose2d[:,0:2], order='uv')
            #     ax2.imshow(target_heatmaps[i])
            #     plt.show()

            return img, target_heatmaps, pose2d, pose3d
        
        else: # for inference
            target_heatmaps = self.heatmap_generator(joints)
            # show
            # for i in range(21):
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(121)
            #     ax2 = fig.add_subplot(122)
            #     print(joints[i])
            #     print(np.argmax(target_heatmaps[i]) - np.argmax(target_heatmaps[i]) // 64 * 64, np.argmax(target_heatmaps[i]) // 64)
            #     ax1.imshow(img)
            #     plot_hand(ax1, joints[:,0:2], order='uv')
            #     plot_hand(ax2, joints[:,0:2], order='uv')
            #     ax2.imshow(target_heatmaps[i])
            #     plt.show()

            img = img.astype(np.float32)
            return np.transpose(img,(2,0,1)), target_heatmaps, pose2d, pose3d
