import os
import shutil
import time
import random
from .FreiHandDataset import FreiHandDataset
from PIL import Image

import cv2
import numpy as np
import pickle
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.frei_utils.fh_utils import *

class FreiHandDataset_Keypoint(FreiHandDataset):
    def __init__(self, config, data_dir, heatmap_generator, transforms=None):
        r"""Read RGB images and their ground truth heat maps
        params:
        @ data_dir: the name of the training or validation data directory
        @ mode: indicates whether the this is a training set ot a test set
        @ transform: data augmentation is implemented if transformers are given
        """
        super().__init__(config.DATA_DIR,
                        data_dir,
                        config.DATASET.DATA_FORMAT)

        assert config.DATASET.NUM_JOINTS == 21, 'Number of joint for RHD is 21!'

        self.config = config
        self.bg_dir = config.DATASET.BACKGROUND_DIR

        if config.DATASET.BACKGROUND_DIR:
            self.bg_list = os.listdir(self.bg_dir)
            
        self.num_joints = config.DATASET.NUM_JOINTS
        self.image_size = np.array(config.MODEL.IMAGE_SIZE)
        self.heatmap_generator = heatmap_generator
        self.transforms = transforms
        self.scale_factor = config.DATASET.SCALE_FACTOR
        self.rotation_factor = config.DATASET.ROT_FACTOR
        self.flip = config.DATASET.FLIP

    def __len__(self):
        return len(self.sample_lst)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        fmt_str += '    Background Image Directory: {}\n'.format(self.bg_dir)

        return fmt_str


    def __getitem__(self, idx):
        # pose2d: [u,v,vis] size B x 21 x 3
        ori_img, msk, pose2d = super().__getitem__(idx)

        # if self.bg_dir:
        #     random_bg = cv2.imread(
        #         os.path.join(self.bg_dir,random.choice(self.bg_list)),
        #         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        #     bg = cv2.cvtColor(cv2.resize(random_bg, img.shape[0:2]),cv2.COLOR_BGR2RGB)
        #     hand = cv2.bitwise_and(msk, img)
            
        #     for i in range(img.shape[0]):
        #         for j in range(img.shape[1]):
        #             for k in range(3):
        #                 if hand[i][j][k] != 0:
        #                     bg[i][j][k] = hand[i][j][k]
            
            # cv2.imshow('1',bg)
            # cv2.waitKey(0)

        if self.transforms: # for training and validation
            img, pose2d = self.transforms( # pose2d is a list, img size: 3 x H x W
                ori_img, [pose2d]
            )
            pose2d = pose2d[0]

            target_heatmaps = self.heatmap_generator(pose2d) # numpy array of size 21 x 64 x 64

            # show
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
            #     plt.title('FreiHand: {} Joint id: {} Vis: {}'.format(idx, k, pose2d[k,2]==1))
            #     plt.show()

            return img, target_heatmaps, pose2d
        
        else: # for inference
            target_heatmaps = self.heatmap_generator(pose2d)
            # show
            # for i in range(21):
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(121)
            #     ax2 = fig.add_subplot(122)
            #     print(pose2d[i])
            #     print(np.argmax(target_heatmaps[i]) - np.argmax(target_heatmaps[i]) // 64 * 64, np.argmax(target_heatmaps[i]) // 64)
            #     ax1.imshow(img)
            #     plot_hand(ax1, pose2d[:,0:2], order='uv')
            #     plot_hand(ax2, pose2d[:,0:2], order='uv')
            #     ax2.imshow(target_heatmaps[i])
            #     plt.show()

            img = img.astype(np.float32)
            return np.transpose(img,(2,0,1)), target_heatmaps, pose2d, img_path

