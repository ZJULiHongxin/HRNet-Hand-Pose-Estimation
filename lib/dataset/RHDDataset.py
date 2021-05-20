import os
import shutil
import time
import logging
from collections import defaultdict
from collections import OrderedDict

from PIL import Image
import matplotlib.pyplot as plt
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

from .standard_legends import idx_RHD

logger = logging.getLogger(__name__)

class RHDDataset(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where dataset is located to.
        dataset (string): Dataset name(train2017, val2017, test2017).
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, subset, data_format=None, transforms=None,
                 target_transform=None):
        # subset: 'training' or 'evaluation'
        self.name = 'RHD'
        self.ori_img_size = (320,320)
        self.data_dir = os.path.join(root, self.name, subset) # ../data/RHD/training or RHD/evaluation

        self.data_format = data_format
        self.transform = transforms
        self.target_transform = target_transform
        self.anno2d_path = os.path.join(self.data_dir, 'anno_%s.pickle' % subset)
        
        self.images = sorted(os.listdir(os.path.join(self.data_dir, 'color')))
        with open(self.anno2d_path, 'rb') as f:
            self.anno_all = pickle.load(f)  # type: dict

        self.reorder_idx = idx_RHD
        self.img_size = 256
        self.hm_size = 64

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_path = os.path.join(self.data_dir, 'color', self.images[idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 320 x 320 x 3
        
        kp_coord_uv = self.anno_all[idx]['uv_vis'][:, :2]  # u, v coordinates of 42 hand key points, pixel
        kp_visible = self.anno_all[idx]['uv_vis'][:, 2:] == 1  # visibility of the key points, boolean

        # choose the one which has the most visible keypoints
        num_visible_kpts_left, num_visible_kpts_right = np.sum(kp_visible[0:21]), np.sum(kp_visible[21:42])
        
        if num_visible_kpts_left >= num_visible_kpts_right:
            pose2d = kp_coord_uv[0:21,:]
            one_hand_visibility = kp_visible[0:21,:]
        else:
            pose2d = kp_coord_uv[21:42,:]
            one_hand_visibility = kp_visible[21:42,:]

        # crop the hand
        one_hand_kp_x, one_hand_kp_y = pose2d[:, 0], pose2d[:, 1]
        leftmost, rightmost = np.min(one_hand_kp_x), np.max(one_hand_kp_x)
        bottommost, topmost = np.max(one_hand_kp_y), np.min(one_hand_kp_y)
        w, h = rightmost - leftmost, bottommost - topmost

        crop_size = min(img.shape[1], int(2 * w if w > h else 2 * h))

        # top_left_corner of the cropped area: [u, v] in u-v image system ↓→
        top_left_corner = [max(0, min(int(leftmost - (crop_size - w) / 2), img.shape[0] - crop_size)),
                        max(0, min(img.shape[1] - crop_size, int(topmost - (crop_size - h) / 2)))]

        # top_left_corner[0]: The distance of the left border from the left border of the original image
        # top_left_corner[1]: The top border from the top border of the original image
        # top_left_corner[0] + crop_size: The distance of the right border from the left border of the original image
        # top_left_corner[1] + crop_size: The distance of the bottom border from the top border of the original image
        cropped_img = orig_img[top_left_corner[1]:top_left_corner[1] + crop_size,\
            top_left_corner[0]:top_left_corner[0] + crop_size, :] # H x W x 3
        # test
        # fig = plt.figure()
        # plt.imshow(img)
        # plt.plot([top_left_corner[0], top_left_corner[0] + crop_size],\
        #     [top_left_corner[1], top_left_corner[1] + crop_size], 'b')
        # plt.title(img_path)
        # plt.show()

        # calculate the ground truth positions of key points on the cropped image
        pose2d = pose2d - np.array(top_left_corner)
        
        if self.transform is not None: # for evaluation
            cropped_img, pose2d = self.transform(cropped_img, [pose2d]) # img: 3 x 256 x 256; pose2d 64
            pose2d = pose2d[0]

        ret = {
            'orig_imgs': orig_img,
            'imgs': cropped_img,
            'pose2d': pose2d[self.reorder_idx],
            'visibility': one_hand_visibility,
            'corner': np.array(top_left_corner),
            'crop_size': crop_size
        }
        
        return ret

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

