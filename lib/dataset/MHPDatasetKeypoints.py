import os
import shutil
import time
import random
from .MHPDataset import MHPDataset
from PIL import Image

import cv2
import numpy as np
import pickle
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.frei_utils.fh_utils import *

class MHPDataset_keypoint(MHPDataset):
    def __init__(self, config, set_name, heatmap_generator, transform=None):
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

        self.config = config
            
        self.num_joints = config.DATASET.NUM_JOINTS
        self.image_size = np.array(config.MODEL.IMAGE_SIZE)
        self.heatmap_generator = heatmap_generator
        self.transform = transform

    def __len__(self):
        return super().__len__()
    
    def __repr__(self):
        fmt_str = '{} Dataset '.format(self.set_name.title()) + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)

        return fmt_str
    
    def __getitem__(self, idx):
        # img: 3 x H x W
        # pose2d: 21 x 3 [u,v,vis]
        # pose3d: 21 x 3 [Xw,Yw,Zw]]
        item = super().__getitem__(idx)

        if self.transform: # for training and validation
            img, pose2d = self.transform( # joints is a list, img size: 3 x H x W
                item['imgs'], [item['pose2d']]
            )
            pose2d = pose2d[0]

            target_heatmaps = self.heatmap_generator(np.concatenate((pose2d, item['visibility']), axis=1))

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
            #     plt.title('MHP: {} Joint id: {} Vis: {}'.format(idx, k, pose2d[k,2]==1))
            #     plt.show()

            ret = {
                'imgs': img,
                'heatmaps': target_heatmaps,
                'pose2d': pose2d,
                'visibility': item['visibility'],
            }
            return ret
        
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
