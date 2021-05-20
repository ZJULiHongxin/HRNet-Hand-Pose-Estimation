from .RHDDataset import RHDDataset
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import kornia
from dataset.frei_utils.fh_utils import *

class RHDDataset_Keypoint(RHDDataset):
    def __init__(self, config, set_name, heatmap_generator, transforms=None):
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
        self.heatmap_generator = heatmap_generator
        self.transforms = transforms
        self.img_size = config.MODEL.IMAGE_SIZE[0]
        self.hm_size = config.MODEL.HEATMAP_SIZE[0]


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
        item = super().__getitem__(idx)
        # orig_img: 320 x 320 x 3
        # cropped_img: H x W x 3
        # pose2d: 21 x 2 in the cropped_mg coordinate system
        # visibility: 21 x 1

        if self.transforms: # for training and validation
            img, pose2d = self.transforms( # pose2d is a list, img (torch.tensor): 3 x 256 x 256
                item['imgs'], [item['pose2d']]
            )

            pose2d = pose2d[0]
            target_heatmaps = self.heatmap_generator(np.concatenate((pose2d, item['visibility']), axis=1))

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
            #     plt.title('RHD: {} Joint id: {} Vis: {}'.format(idx, k, pose2d[k,2]==1))
            #     plt.show()
            ret = {
                'orig_imgs': item['orig_imgs'],
                'imgs': img,
                'pose2d': pose2d,
                'heatmaps': target_heatmaps,
                'visibility': item['visibility'],
                'corner': item['corner'],
                'crop_size': item['crop_size']
            }
            return ret
        
        else: # for inference
            target_heatmaps = self.heatmap_generator(pose2d)
            img = img.astype(np.float32)
            return np.transpose(img,(2,0,1)), target_heatmaps, pose2d, img_path

