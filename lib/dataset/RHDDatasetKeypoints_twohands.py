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

    # one hand anno
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

    # # two hand anno
    # def __getitem__(self, idx):
    #     two_hand_img, two_hand_joints, meta = super().__getitem__(idx)
    #     down_scale = self.config.MODEL.HEATMAP_SIZE[0] / self.config.MODEL.IMAGE_SIZE[0]
    #     # read the 2D keypoints annotation of this image. Note: "self.anno_all[idx]['uv_vis']"" is a numpy array of size 42x3.
    #     # The first colum of 2D kp annotation matrix represents the x-axis (horizontal and positive rightwards) value of a key point;
    #     # the second column represents the y-axis (vertical and positive upwards) value of a key point;
    #     # the third column consists boolean values donoting the visibility of key points (1 for visible points and 0 otherwise)
    #     if self.transforms: # for training and validation
    #         if meta['imgnum'] == 2:
                
    #             print(two_hand_joints[0:21,:])
    #             left_hand_img, left_hand_joints = self.transforms(    # left_hand_img is a torch tensor whose size: 3x 256(H) x 256(W)
    #                 two_hand_img[:,:,0:3], [two_hand_joints[0:21,:]]  # left_hand_joints is a one-element list, numpy array, size 21 x 3
    #             )
    #             print(left_hand_joints[0])
    #             input()
    #             right_hand_img, right_hand_joints = self.transforms( # joints is a list, img size: H x W x 3
    #                 two_hand_img[:,:,3:6], [two_hand_joints[21:42,:]]
    #             )

    #             left_hand_joints, right_hand_joints = down_scale * left_hand_joints[0], down_scale * right_hand_joints[0]
    #             left_hand_target_heatmaps = self.heatmap_generator(left_hand_joints) # left_hand_target_heatmaps is a numpy array whose size = 21 x 64(H) x 64(W)
    #             right_hand_target_heatmaps = self.heatmap_generator(right_hand_joints)

    #             return  torch.cat((left_hand_img, right_hand_img), dim=0), \
    #                     np.concatenate((left_hand_target_heatmaps, right_hand_target_heatmaps), axis=0), \
    #                     np.concatenate((left_hand_joints, right_hand_joints), axis=0), \
    #                     meta
    #         elif meta['imgnum'] == 1:
    #             one_hand_img, one_hand_joints = self.transforms(    # left_hand_img is a torch tensor whose size: 3x 256(H) x 256(W)
    #                 two_hand_img, [two_hand_joints]  # left_hand_joints is a one-element list, numpy array, size 21 x 3
    #             )
    #             one_hand_joints = down_scale * one_hand_joints[0]
    #             one_hand_target_heatmaps = self.heatmap_generator(one_hand_joints) # left_hand_target_heatmaps is a numpy array whose size = 21 x 64(H) x 64(W)

    #             return  one_hand_img, \
    #                     one_hand_target_heatmaps, \
    #                     one_hand_joints, \
    #                     meta
    #     else: # for inference
    #         if meta['imgnum'] == 2:
    #             two_hand_joints = down_scale * two_hand_joints
    #             left_hand_target_heatmaps = self.heatmap_generator(two_hand_joints[0:21,:]) # size = 21 x 64(H) x 64(W)
    #             right_hand_target_heatmaps = self.heatmap_generator(two_hand_joints[21:42,:])
    #             img = img.astype(np.float32)
    #             return two_hand_img, np.concatenate((left_hand_target_heatmaps, right_hand_target_heatmaps), axis=0), two_hand_joints, meta
    #         elif meta['imgnum'] == 1:
    #             one_hand_joints = down_scale * two_hand_joints
    #             one_hand_target_heatmaps = self.heatmap_generator(one_hand_joints)
    #             img = img.astype(np.float32)
    #             return two_hand_img, one_hand_target_heatmaps, one_hand_joints, meta


    def visualize_samples(config, data_loader):
    # visualize some samples
        for two_hand_img, two_hand_target_heatmaps, two_hand_joints, meta in data_loader:
            # batch has three keys:
            # key 1: batch['image']         size: batch_size x 3(C) x 256(W) x 256(H)
            # key 2: batch['heat_map_gn']   size: batch_size x 21(kps) x 64(W) x 64(H)
            # each of the 21 heat maps is of size 64(W) x 64(H)
            # key 3: batch['2d_kp_anno']     size: batch_size x 21(kps) x 2 [u,v]
            # The first row contains u-axis coordinates and the second row contains v-axis values
            img = img = cv2.imread(meta['image_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img,(2,0,1))
            
            for i in range(0,21,5):
                fig = plt.figure(1)
                ax1 = fig.add_subplot(1,5,1)
                ax2 = fig.add_subplot(1,5,2)
                ax3 = fig.add_subplot(1,5,3)
                ax2 = fig.add_subplot(1,5,4)
                ax3 = fig.add_subplot(1,5,5)
                ax1.imshow(img)  # the original image
                
                lh, rh = two_hand_img[0:3,:,:], two_hand_img[3:6,:,:]
                lh_joints, rh_joints = two_hand_joints[0:21,:], two_hand_joints[21:42,:]
                lh_hm, rh_hm = two_hand_target_heatmaps[0:21,:], two_hand_target_heatmaps[21:42,:]
                lh_corner, rh_corner = meta['tl_coner']
                lh_crop_size, rh_crop_size = meta['crop_size']

                ax1.plot(lh_joints[i,0], lh_joints[i,1], 'r*')
                ax1.plot([lh_corner[0], lh_corner[0] + lh_crop_size],
                        [lh_corner[1], lh_corner[1] + lh_crop_size], 'r')  # the cropped area
            
                cropped_img_tensor = cv2.resize(lh, self.config.HEATMAP_SIZE)
                ax2.imshow(cropped_img_tensor)
                ax2.plot(lh_joints[i,0], lh_joints[i,1], 'r*')
                ax3.imshow(lh_hm[i])

                cropped_img_tensor = cv2.resize(rh, self.config.HEATMAP_SIZE)
                ax4.imshow(cropped_img_tensor)
                ax4.plot(rh_joints[i,0], rh_joints[i,1], 'r*')
                ax5.imshow(rh_hm[i])

                plt.show()

