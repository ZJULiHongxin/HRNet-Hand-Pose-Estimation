import os
import shutil
import time
import random
from .FHADataset import FHADataset
from PIL import Image

import cv2
import numpy as np
import pickle
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .FHA_utils import *
from .HandGraph_utils.vis import *

class FHADataset_Keypoint(FHADataset):
    def __init__(self, config, data_dir, heatmap_generator, transforms=None):
        r"""Read RGB images and their ground truth heat maps
        params:
        @ data_dir: the name of the training or validation data directory
        @ mode: indicates whether the this is a training set ot a test set
        @ transform: data augmentation is implemented if transformers are given
        """
        super().__init__(
                        config,
                        data_dir,
                        config.DATASET.DATA_FORMAT)

        assert config.DATASET.NUM_JOINTS == 21, 'Number of joint for FHA is 21!'

        self.config = config
        self.NFrames = config.DATASET.N_FRAMES
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
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def __getitem__(self, idx):
        # imgs: H x W x N_frames * 3
        # pose2d_gts: N_frames * 21 x 2 [u,v,vis]
        # pose3d_gts: N_frames x 21 x 3 (cam coord)
        imgs, pose2d_gts, pose3d_gts = super().__getitem__(idx)

        if self.transforms: # for training and validation
            imgs, pose2d_gts = self.transforms( # joints is a list, img size: 3 x H x W
                imgs, [pose2d_gts]
            )
            pose2d_gts = pose2d_gts[0]

            # visibility
            s = self.config.MODEL.HEATMAP_SIZE[0] # 64
            for k in range(pose2d_gts.shape[0]):
                if pose2d_gts[k,0] >= s or pose2d_gts[k,0] < 0 or pose2d_gts[k,1] >= s or pose2d_gts[k,1] < 0:
                    pose2d_gts[k,2] = 0
            

            target_heatmaps = self.heatmap_generator(pose2d_gts) # numpy array of size 21 x 64 x 64

            # show
            # for i in range(21):
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(121)
                
            #     print(pose2d_gts[0:21])
            #     print(np.argmax(target_heatmaps[i]) - np.argmax(target_heatmaps[i]) // 64 * 64, np.argmax(target_heatmaps[i]) // 64)
            #     ax1.imshow(np.transpose(imgs[0:3].numpy(),(1,2,0)))
            #     ax1.scatter(4*pose2d_gts[i,0],4*pose2d_gts[i,1],linewidths=1)

            #     ax2 = fig.add_subplot(122)
            #     ax2.imshow(target_heatmaps[i])
            #     plt.show()

            return imgs.view(imgs.shape[0] // 3, 3, imgs.shape[1], imgs.shape[2]), \
                    target_heatmaps.reshape(target_heatmaps.shape[0] // 21, 21, target_heatmaps.shape[1],target_heatmaps.shape[2]), \
                    pose2d_gts.reshape(pose2d_gts.shape[0] // 21, 21, 3), \
                    pose3d_gts
        
        else: # for inference
            target_heatmaps = self.heatmap_generator(pose2d_gts)
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

            imgs = imgs.astype(np.float32)
            return np.transpose(imgs,(2,0,1)), target_heatmaps, joints

    def visualize_data(self):
        for i in range(self.__len__()):
            # img (torch.tensor): N_frames x 3 x H x W
            # target_heatmaps: N_frames x 21 x 64(H) x 64(W)
            # pose2d_gts: N_frames x 21 x 3 [u,v,vis]
            # pose3d_gts: N_frames x 21 x 3 (cam coord)
            imgs, target_heatmaps, pose2d_gts, pose3d_gts, img_path_list = self.__getitem__(i)

            im_height, im_width = imgs.shape[:2]

            for frame_id in range(imgs.shape[0]):
                img = cv2.cvtColor(np.transpose(imgs[frame_id].numpy(),(1,2,0)), cv2.COLOR_RGB2BGR)
                pose2d = 4 * pose2d_gts[frame_id]
                pose3d = pose3d_gts[frame_id]
                pose3d = pose3d - pose3d[0]
                print(pose3d)

                fig = plt.figure(frame_id)
                # fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

                # 1. plot raw image
                ax = fig.add_subplot(1,2,1)  
                ax.imshow(img)
                ax.set_title(img_path_list[frame_id])
                
                for i in range(21):
                    if pose2d[i,2] == 0:
                        plt.scatter(pose2d[i,0],pose2d[i,1])
                visualize_joints_2d(ax, pose2d, joint_idxs=False)

                # 3. plot 3D joints
                ax = fig.add_subplot(122, projection='3d')
                draw_3d_skeleton_on_ax(pose3d, ax)
                ax.set_title("GT 3D joints")

            plt.show()
    