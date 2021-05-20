# -*-coding:UTF-8-*-
import os
import re
import torch
import scipy.io
import pickle
import numpy as np
import glob
import fnmatch
import torch.utils.data as data
import scipy.misc
from PIL import Image
import cv2
from .transforms import Mytransforms
from .standard_legends import std_legend_lst, idx_MHP
from dataset.frei_utils.fh_utils import plot_hand

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def recursive_glob(rootdir='.', pattern='*'):
	matches = []
	for root, dirnames, filenames in os.walk(rootdir):
	  for filename in fnmatch.filter(filenames, pattern):
		  matches.append(os.path.join(root, filename))

	return matches

def readAnnotation3D(file):
	f = open(file, "r")
	an = []
	for l in f:
		l = l.split()
		an.append((float(l[1]),float(l[2]), float(l[3])))

	return np.array(an, dtype=float)
def read_mat_file(mode, root_dir, img_list):
    """
        get the groundtruth

        mode (str): 'lsp' or 'lspet'
        return: three list: key_points list , centers list and scales list

        Notice:
            lsp_dataset differ from lspet dataset
    """
    mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
    # lspnet (14,3,10000)
    if mode == 'lspet':
        lms = mat_arr.transpose([2, 1, 0])
        pose2ds = mat_arr.transpose([2, 0, 1]).tolist()
    # lsp (3,14,2000)
    if mode == 'lsp':
        mat_arr[2] = np.logical_not(mat_arr[2])
        lms = mat_arr.transpose([2, 0, 1])
        pose2ds = mat_arr.transpose([2, 1, 0]).tolist()

    centers = []
    scales = []
    for idx in range(lms.shape[0]):
        im = Image.open(img_list[idx])
        w = im.size[0]
        h = im.size[1]
        # lsp and lspet dataset doesn't exist groundtruth of center points
        center_x = (lms[idx][0][lms[idx][0] < w].max() +
                    lms[idx][0][lms[idx][0] > 0].min()) / 2
        center_y = (lms[idx][1][lms[idx][1] < h].max() +
                    lms[idx][1][lms[idx][1] > 0].min()) / 2
        centers.append([center_x, center_y])

        scale = (lms[idx][1][lms[idx][1] < h].max() -
                lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
        scales.append(scale)

    return pose2ds, centers, scales


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class MHP_CPMDataset(data.Dataset):
    """
        Args:
            root_dir (str): the path of train_val dateset.
            stride (float): default = 8
            transformer (Mytransforms): expand dataset.
        Notice:
            you have to change code to fit your own dataset except LSP

    """

    def __init__(self, config, set_name, heatmap_generator=None, transform=None, stride=8):
        self.exception = False
        self.name = 'MHP'
        self.config = config
        self.orig_img_size = [640, 480]
        self.data_dir = os.path.join(config.DATA_DIR, self.name) # FreiHAND
        self.image_paths = recursive_glob(self.data_dir, "*_webcam_[0-9]*")
        self.image_paths = natural_sort(self.image_paths)
        self.set_name = set_name
        self.split = 0.8 # According to the dataset paper, the 20% for the test split and the remaining 80% for the training split.

        if set_name in ['train', 'training']:
            self.start_idx = 0
            self.end_idx = int(len(self.image_paths) * self.split)
            self.transform = Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(256),
                Mytransforms.RandomHorizontalFlip(),
            ])
        elif set_name in ['eval', 'valid', 'val', 'evaluation', 'validation']:
            self.start_idx = int(len(self.image_paths) * self.split)
            self.end_idx = len(self.image_paths)
            self.transform = Mytransforms.Compose([Mytransforms.TestResized(256)])

        Fx, Fy, Cx, Cy = 614.878, 615.479, 313.219, 231.288

        self.intrinsic_matrix = np.array([[Fx, 0, Cx],
                                          [0, Fy, Cy],
                                          [0,  0, 1 ]])

        self.distortion_coeffs = np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])
        
        # rearrange the order of the annotations of 21 joints
        self.reorder_idx = idx_MHP
        self.stride = stride
        self.sigma = config.DATASET.SIGMA
 
        

    def __getitem__(self, idx):
        self.exception = False
        img_path = self.image_paths[self.start_idx + idx]
        img = cv2.resize(cv2.imread(img_path), tuple(self.config.MODEL.IMAGE_SIZE))

        # load 3D pose (world coord)
        dir_name, img_name = os.path.split(img_path) # ex: ../MHP/annotated_frames/data_1, 0_webcam_1.jpg
        dir_id = dir_name.split('_')[-1]
        img_idx, _, webcam_id = img_name[0:-4].split('_')

        # ex: ../MHP/annotated_frames/data1/0_joints.txt
        pose3d_path = os.path.join(self.data_dir, 'annotations', os.path.basename(dir_name), img_idx + '_joints.txt')
        pose3d = readAnnotation3D(pose3d_path)[self.reorder_idx]

        # load extrinsic params
        rvec = pickle.load(
            open(
                os.path.join(
                    self.data_dir, 'calibrations', 'data_{}'.format(dir_id), 'webcam_{}'.format(webcam_id), 'rvec.pkl'), "rb"), encoding='latin1')
        tvec = pickle.load(open(os.path.join(self.data_dir, 'calibrations', 'data_{}'.format(dir_id), 'webcam_{}'.format(webcam_id), 'tvec.pkl'), "rb"), encoding='latin1')

        pose2d, _ = cv2.projectPoints(pose3d, rvec, tvec, self.intrinsic_matrix, self.distortion_coeffs) # 21 x 1 x 2
        pose2d = pose2d.squeeze() # 21 x 2
        pose2d[:,0] *= (256 / self.orig_img_size[0])
        pose2d[:,1] *= (256 / self.orig_img_size[1]) 

        h, w = img.shape[0:2]

        visibility = np.ones((pose2d.shape[0],1))
        for k in range(pose2d.shape[0]):
            if pose2d[k,0] < 0 or pose2d[k,1] < 0 or pose2d[k,0] >= w or pose2d[k,1] >= h:
                visibility[k] = 0
        pose2d = np.concatenate((pose2d, visibility), axis=1)

        try:
            center_x = (pose2d[np.argwhere(pose2d[:,0] < w), 0].max() +
                        pose2d[np.argwhere(pose2d[:,0] > 0), 0].min()) / 2
        except:
            center_x = w / 2
            self.exception = True
        try:
            center_y = (pose2d[np.argwhere(pose2d[:,1] < h), 1].max() +
                        pose2d[np.argwhere(pose2d[:,1] > 0), 1].min()) / 2
        except:
            center_y = h / 2
            self.exception = True
        center = [center_x, center_y]

        try:
            scale = (pose2d[np.argwhere(pose2d[:,1] < h), 1].max() -
                    pose2d[np.argwhere(pose2d[:,1] > 0), 1].min() + 4) / h
        except:
            scale = 0.5
        # expand dataset. pose2d 21 x 3 [u,v,vis]
        img, pose2d, center = self.transform(img, pose2d, center, scale)

        heatmap = np.zeros((len(pose2d) + 1, h // self.stride, w // self.stride), dtype=np.float32)
        for i in range(len(pose2d)):
            # resize from 256 to 32
            x = int(pose2d[i][0]) * 1.0 / self.stride
            y = int(pose2d[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=h / self.stride, size_w=w / self.stride, center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[i + 1, :, :] = heat_map

        heatmap[0, :, :] = 1.0 - np.max(heatmap[1:, :, :], axis=0)  # for background

        # show
        # import matplotlib.pyplot as plt
        
        # for k in range(0,21,1):
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(121)
        #     ax2 = fig.add_subplot(122)
        #     print('subpixel:',pose2d[k])
        #     ax1.imshow(cv2.cvtColor(img / img.max(), cv2.COLOR_BGR2RGB))
        #     plot_hand(ax1, pose2d[:,0:2], order='uv')
        #     ax2.imshow(heatmap[k])
        #     plot_hand(ax2, pose2d[:,0:2] / self.stride, order='uv')
        #     plt.title('MHP: {} Joint id: {} Vis: {}'.format(idx, k, pose2d[k,2]==1))
        #     plt.show()

        centermap = np.zeros((h, w, 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=h, size_w=w, center_x=center[0], center_y=center[1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])

        centermap = Mytransforms.to_tensor(centermap)

        ret = {
        'imgs': img, # 3 x 256 x 256
        'pose2d': pose2d[:,0:-1] / self.stride,
        'heatmaps': heatmap, # (21+1) x 32 x 32
        'visibility': visibility,
        'centermaps': centermap # 1 x 256 x 256
        }

        return ret
    def __repr__(self):
        fmt_str = '{} Dataset '.format(self.set_name.title()) + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        
        return fmt_str

    def __len__(self):
        return self.end_idx - self.start_idx

    def get_kpts(self, maps, img_h = 256.0, img_w = 256.0):
        # maps (b,21,32,32)
        pose2d_pred_u =  torch.argmax(maps.view((maps.shape[0], maps.shape[1],-1)),dim=2)
        pose2d_pred_u, pose2d_pred_v = pose2d_pred_u % maps.shape[-1], pose2d_pred_u // maps.shape[-1]

        return torch.stack((pose2d_pred_u, pose2d_pred_v), dim=2).float()
