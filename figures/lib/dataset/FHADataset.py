import os
import shutil
import time
import logging
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from .standard_legends import std_legend_lst, idx_FHA

logger = logging.getLogger(__name__)

class FHADataset(Dataset):
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

    def __init__(self, config, set_name, data_format, transform=None,
                 target_transform=None):
        self.name = 'FHA'
        self.root = os.path.join(config.DATASET.ROOT, 'Videos')
        self.set_name = set_name
        self.NFrames = config.DATASET.N_FRAMES
        # Subjects 1-5 for training and 6 for evaluation 
        self.subjects = ['Subject_%d' % i for i in range(1,2)] if 'train' in set_name  else ['Subject_2']

        self.cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594,
            25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
            [-0.000969709653873, 0.00274303671904, 0.99999576807,
            3.902], [0, 0, 0, 1]])
        self.cam_intr = np.array([[1395.749023, 0, 935.732544],
                            [0, 1395.749268, 540.681030], [0, 0, 1]])     
        
        self.cur_sub_idx = 0

        self.cur_action_list = os.listdir(os.path.join(self.root, self.subjects[self.cur_sub_idx]))
        self.cur_action_idx = 0

        self.cur_video_idx = 0
        self.cur_video_list = os.listdir(os.path.join(
            self.root,
            self.subjects[self.cur_sub_idx],
            self.cur_action_list[self.cur_action_idx]))

        self.video_dir = os.path.join(
            self.root,
            self.subjects[self.cur_sub_idx],
            self.cur_action_list[self.cur_action_idx],
            self.cur_video_list[self.cur_video_idx])

        self.frames_dir = os.path.join(self.video_dir, 'color')

        self.cur_frame_idx = 0
        self.stride = config.DATASET.FRAME_STRIDE

        self.skel_root = os.path.join(config.DATASET.ROOT, 'Hand_pose_annotation_v1')
        cur_skeleton_path = os.path.join(
            self.skel_root,
            self.subjects[self.cur_sub_idx],
            self.cur_action_list[self.cur_action_idx],
            self.cur_video_list[self.cur_video_idx],
            'skeleton.txt')

        self.skeleton_vals = np.loadtxt(cur_skeleton_path)

        self.transform = transform
        self.target_transform = target_transform

        legend_dict = OrderedDict(sorted(zip(std_legend_lst, idx_FHA), key=lambda x:x[1]))
        self.joint_label_list = list(legend_dict.keys())

        # rearrange the order of the annotations of 21 joints
        self.reorder_idx = np.array([
            0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
            20
        ])
    
    def update(self):
        if self.cur_frame_idx + self.stride * (self.NFrames - 1) != len(os.listdir(self.frames_dir)) - 1: 
            self.cur_frame_idx += 1
        else:
            self.cur_frame_idx = 0
            if self.cur_video_idx != len(self.cur_video_list) - 1:             
                self.cur_video_idx += 1
            else:
                self.cur_video_idx = 0
                if self.cur_action_idx != len(self.cur_action_list) - 1:
                    self.cur_action_idx += 1
                else:
                    self.cur_action_idx = 0
                    if self.cur_sub_idx != len(self.subjects) - 1:
                        self.cur_sub_idx += 1
                        self.cur_action_list = os.listdir(os.path.join(
                            self.root,
                            self.subjects[self.cur_sub_idx]))
                    else:
                        print('Run out of samples')
                
                self.cur_video_list = os.listdir(os.path.join(
                        self.root,
                        self.subjects[self.cur_sub_idx],
                        self.cur_action_list[self.cur_action_idx]))

            frame_dir = os.path.join(
                self.subjects[self.cur_sub_idx],
                self.cur_action_list[self.cur_action_idx],
                self.cur_video_list[self.cur_video_idx])

            self.frames_dir = os.path.join(
                self.root,
                frame_dir,
                'color')
            skel_path = os.path.join(
                self.skel_root,
                frame_dir,
                'skeleton.txt')

            self.skeleton_vals = np.loadtxt(skel_path)
            #print('Loading skeleton from {}'.format(skel_path))

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        # Read a sequence of length NFrames
        multi_frame = []
        multi_pose_gt = []
        img_path_list = []

        for i in range(self.cur_frame_idx, self.cur_frame_idx + self.stride * self.NFrames, self.stride):
            pose3d_gt = self.skeleton_vals[:, 1:].reshape(self.skeleton_vals.shape[0], 21, -1)[i][self.reorder_idx]

            multi_pose_gt.append(pose3d_gt)

            img_path = os.path.join(self.frames_dir, 'color_%04d.jpeg' % i)
            img_path_list.append(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            multi_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        frames = np.concatenate(multi_frame, axis=2)

        pose3d_gts = np.concatenate(multi_pose_gt, axis=0)
        pose3d_gts_homo = np.concatenate([pose3d_gts, np.ones([pose3d_gts.shape[0], 1])], 1) # N x 4
        pose3d_gts_camcoords = self.cam_extr.dot(
                pose3d_gts_homo.transpose()).transpose()[:, :3].astype(np.float32) # N x 3

        pose2d_gts_homo = np.array(self.cam_intr).dot(pose3d_gts_camcoords.transpose()).transpose() # N x 3
        pose2d_gts = (pose2d_gts_homo / pose2d_gts_homo[:, 2:])[:, :2]

        # located at the next sequence
        self.update()

        # # show
        # fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # ax1.imshow(img)
        # plot_hand(ax1, uv, order='uv')
        # plot_hand(ax2, uv, order='uv')
        # ax1.axis('off')
        # ax2.axis('off')
        # plt.show()

        pose2d_gts = np.concatenate((pose2d_gts, np.ones((pose2d_gts.shape[0],1))), axis=1) # visibility
        
        # visibility
        for k in range(pose2d_gts.shape[0]):
            if pose2d_gts[k,0] >= 1920 or pose2d_gts[k,0] < 0 or pose2d_gts[k,1] >= 1080 or pose2d_gts[k,1] < 0:
                pose2d_gts[k,2] = 0

        if self.transform is not None:
            frames, pose2d_gts_list = self.transform(frames,[pose2d_gts])
            
            # frames: N_frames x 3 x H x W
            # pose2d_gts: N_frames*21 x 3 [u,v,vis]
            # pose3d_gts: N_frames x 21 x 3 [x,y,z] (cam coord)
            return frames.view(frames.shape[0] // 3, 3, frames.shape[1], frames.shape[2]), \
            pose2d_gts_list[0], pose3d_gts_camcoords.reshape([pose3d_gts_camcoords.shape[0] // 21, 21, 3])

        if self.target_transform is not None:
            pose2d_gts = self.target_transform(pose2d_gts)
        
        # frames: H x W x N_frames * 3
        # pose2d_gts: N_frames*21 x 3 [u,v,vis]
        # pose3d_gts: N_frames x 21 x 3 [x,y,z] (cam coord)
        # img_patj_list: a list of size N_frames
        return frames, pose2d_gts, pose3d_gts_camcoords.reshape([pose3d_gts_camcoords.shape[0] // 21, 21, 3])
    

    def __len__(self):
        Nsamples = 0
        for sub in self.subjects:
            action_list = os.listdir(os.path.join(self.root, sub))
            for act in action_list:
                video_list = os.listdir(os.path.join(self.root, sub, act))
                for video in video_list:
                    frames = os.listdir(os.path.join(self.root, sub, act, video, 'color'))
                    n = len(frames) - self.stride * (self.NFrames - 1)
                    Nsamples += n
        return Nsamples


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): ' + 'Using transforms' if self.transform else '    Transforms (if any): None'
        fmt_str += tmp
        #tmp = '    Target Transforms (if any): '
        #fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

