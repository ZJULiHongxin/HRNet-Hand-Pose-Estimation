import os
import pickle
import shutil
import time
import fnmatch
import logging
from collections import defaultdict, OrderedDict
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from .standard_legends import std_legend_lst, idx_MHP
from dataset.frei_utils.fh_utils import load_db_annotation, projectPoints, db_size, plot_hand

from .FHA_utils import *
from .HandGraph_utils.vis import *

logger = logging.getLogger(__name__)

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

class MHPSeqDataset(Dataset):
    """
    Split: 4:1
    """
    def __init__(self, config, set_name, transform=None,
                 heatmap_generator=None):
        self.name = 'MHP'
        self.orig_img_size = [640,480]
        self.transform = transform
        self.heatmap_generator = heatmap_generator
        self.data_dir = os.path.join(config.DATA_DIR, 'MHP') # ../data/MHP
        self.anno_dir = 'annotations'
        # rearrange the order of the annotations of 21 joints
        self.reorder_idx = idx_MHP

        # load samples onto 
        self.rvec_dict, self.tvec_dict, self.data_list, self.pose3d_dict = {}, {}, [], {}
        if set_name in ['train', 'training']:
            subdir_range = range(1,17)
            # self.data_list += [os.path.join('augmented_samples','data_{}'.format(i)) for i in range(1,10)] \
            #     + [os.path.join('augmented_samples','data_11'), os.path.join('augmented_samples','data_14'), os.path.join('augmented_samples','data_15'), os.path.join('augmented_samples','data_16')]
            #self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(1,3)]
        elif set_name in ['eval', 'valid', 'val', 'evaluation', 'validation']:
            subdir_range = range(17,22)
            # self.data_list += [os.path.join('augmented_samples','data_17'), os.path.join('augmented_samples','data_18'), os.path.join('augmented_samples','data_19'), os.path.join('augmented_samples','data_21')]
            #self.data_list = [os.path.join('annotated_frames','data_{}'.format(i)) for i in range(3,5)]
        for i in subdir_range:
            data_subdir = 'data_{}'.format(i)
            if data_subdir not in self.rvec_dict:
                self.rvec_dict[data_subdir] = {}
            if data_subdir not in self.tvec_dict:
                self.tvec_dict[data_subdir] = {}
            if data_subdir not in self.pose3d_dict:
                self.pose3d_dict[data_subdir] = {}
            data_dir = os.path.join(self.data_dir, 'annotated_frames', data_subdir)
            self.data_list.append(data_dir)
            for cam_idx in range(1,5):
                self.rvec_dict[data_subdir][str(cam_idx)] = pickle.load(open(os.path.join(self.data_dir, 'calibrations', data_subdir, 'webcam_{}'.format(cam_idx), 'rvec.pkl'), "rb"), encoding='latin1')
                self.tvec_dict[data_subdir][str(cam_idx)] = pickle.load(open(os.path.join(self.data_dir, 'calibrations', data_subdir, 'webcam_{}'.format(cam_idx), 'tvec.pkl'), "rb"), encoding='latin1')
            for frame_idx in range(len(os.listdir(data_dir)) // 4):
                    # ex: ../MHP/annotattions/data_1/0_joints.txt
                pose3d_path = os.path.join(self.data_dir, self.anno_dir, data_subdir, str(frame_idx) + '_joints.txt')
                self.pose3d_dict[data_subdir][frame_idx] = readAnnotation3D(pose3d_path)[self.reorder_idx] # 21 x 3

        self.cur_data_idx = 0
        self.cur_datadir_len = len(os.listdir(os.path.join(self.data_list[self.cur_data_idx]))) // 4
        self.cur_frame_idx = 0

        Fx, Fy, Cx, Cy = 614.878, 615.479, 313.219, 231.288

        # self.intrinsic_matrix = update_after_resize(
        #     np.array([[Fx, 0, Cx],
        #             [0, Fy, Cy],
        #             [0,  0, 1 ]], dtype='float32'),  # dtype('float64')
        #             image_shape=(480, 640), new_image_shape=self.config.MODEL.HEATMAP_SIZE
        # )
        self.intrinsic_matrix = np.array([[Fx, 0, Cx],
                                            [0, Fy, Cy],
                                            [0,  0, 1 ]], dtype='float32')
                                    
        self.distortion_coeffs = 0 * np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])
        
        # calculate the number of samples
        self.stride = config.DATASET.STRIDE
        Nsamples = 0
        for data in self.data_list:
            frame_list = os.listdir(data)
            Nsamples += (len(frame_list) // 4 - 1) // self.stride + 1
        
        self.l = Nsamples
        self.seq_idx = config.DATASET.SEQ_IDX # -2 -1 0 1 2
        self.seq_len = len(self.seq_idx)
        self.n_views = [1,2,3,4]
        self.last_ret = None

    def __len__(self):
        return self.l

    def update(self):
        self.cur_frame_idx += self.stride

        if self.cur_frame_idx >= self.cur_datadir_len:
            #print('next camera')
            self.cur_frame_idx = 0

            if self.cur_data_idx < len(self.data_list) - 1:
                self.cur_data_idx += 1   
            else:
                # reset the indices for the next epoch
                self.cur_data_idx = 0
                self.cur_frame_idx = 0
                print('Run out of samples') 
            
            self.cur_datadir_len = len(os.listdir(os.path.join(self.data_list[self.cur_data_idx]))) // 4

    def __getitem__(self, i):
        img_lst = []
        orig_img_lst = []
        pose2d_lst = []
        pose3d_lst = []
        heatmap_lst = []
        proj_matrix_lst = []
        visibility_lst = []
        ret = {
            'imgs': [],
            'orig_imgs': [],
            'heatmaps': [],
            'pose2d': [],
            'visibility': [],
            'pose3d': [],
            'extrinsic_matrices': [],
        }
        subdir = os.path.basename(self.data_list[self.cur_data_idx]) #
        seq_len = self.seq_len
 
        for j in range(seq_len): # -2d, -d, 0, d, 2d
            idx = self.seq_idx[j]
            frame_idx = max(0, min(self.cur_datadir_len - 1, self.cur_frame_idx + idx))
            pose3d_gt = self.pose3d_dict[subdir][self.cur_frame_idx].astype('float32') # 21 x 3
            ret['pose3d'].append(pose3d_gt)
            #print(self.cur_frame_idx, idx)
            if j == seq_len - 1 \
                or self.cur_frame_idx + self.seq_idx[0] <= 0 \
                or self.cur_frame_idx + self.seq_idx[-1] >= self.cur_datadir_len - 1:
                for cam_idx in self.n_views:#range(1,1 + self.n_views):
                    img_path = os.path.join(self.data_list[self.cur_data_idx], '{}_webcam_{}.jpg'.format(frame_idx, cam_idx))
                    #print(img_path)
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # 480(H) x 640(W) x 3
                    ret['orig_imgs'].append(img)
                    # load 3D pose (world coord)
                    dir_name, img_name = os.path.split(img_path) # ex: ../MHP/annotated_frames/data_1, 0_webcam_1.jpg
                    
                    # load extrinsic params. rvec: rotation vectors (Rodrigues); tvec: translation vectors
                    webcam_id = img_name[-5]
                    rvec = self.rvec_dict[subdir][webcam_id] 
                    tvec = self.tvec_dict[subdir][webcam_id]
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    rigidMatrix = np.concatenate((np.float32(rotation_matrix), np.float32(tvec).reshape((3,1))), axis=1)
                    #rot_mat_lst.append(rotation_matrix)
                    #tvec_lst.append(tvec)
        
                    # project 3D points (world coord) onto the image plane
                    pose3d_cam = np.dot(rotation_matrix, np.transpose(pose3d_gt, (1,0))) + tvec.reshape((3,1)) # 3 x 21

                    # project 3D points (world coord) onto the image plane
                    zeros = np.array([0.,0.,0.])
                    pose2d, _ = cv2.projectPoints(
                        objectPoints=pose3d_cam,
                        rvec=zeros, tvec=zeros,
                        cameraMatrix=self.intrinsic_matrix,
                        distCoeffs=self.distortion_coeffs) # 21 x 1 x 2
                    pose2d = pose2d.squeeze()

                    visibility = np.ones((pose2d.shape[0],1))
                    img_height, img_width = img.shape[0:2]
                    for k in range(pose2d.shape[0]):
                        if pose2d[k,0] < 0 or pose2d[k,1] < 0 or pose2d[k,0] >= img_width or pose2d[k,1] >= img_height:
                            visibility[k] = 0

                    img, pose2d = self.transform(img,[pose2d])
                    pose2d = pose2d[0]
                    """
                    pack all items up
                    """
                    if self.heatmap_generator is not None:
                        heatmap = self.heatmap_generator(np.concatenate((pose2d, visibility), axis=1))
                        ret['heatmaps'].append(heatmap)

                    ret['imgs'].append(img)
                    ret['extrinsic_matrices'].append(rigidMatrix)
                    ret['pose2d'].append(pose2d)
                    ret['visibility'].append(visibility)

            else:
                ret['imgs'].extend(self.last_ret['imgs'][4 * (j+1): 4 * (j+2)])
                ret['pose2d'].extend(self.last_ret['pose2d'][4 * (j+1): 4 * (j+2)])
                ret['visibility'].extend(self.last_ret['visibility'][4 * (j+1): 4 * (j+2)])
                if self.heatmap_generator is not None:
                    ret['heatmaps'].extend(self.last_ret['heatmaps'][4 * (j+1): 4 * (j+2)])
                ret['extrinsic_matrices'].extend(self.last_ret['extrinsic_matrices'][4 * (j+1): 4 * (j+2)])
                ret['orig_imgs'].extend(self.last_ret['orig_imgs'][4 * (j+1): 4 * (j+2)])
                
        self.update()

        if self.cur_datadir_len - 1 - self.seq_idx[-1] >= self.cur_frame_idx >= -self.seq_idx[0]:
            self.last_ret = copy.deepcopy(ret)
        
        for k in ret.keys():
            if len(ret[k]) > 0:
                ret[k] = torch.stack(ret[k]) if k == 'imgs' else np.stack(ret[k])
            #print(k,ret[k].shape)
        
        ret['intrinsic_matrix'] = self.intrinsic_matrix

        #show
        # if self.cur_frame_idx > 20:
        #     pose2d = ret['pose2d'][0]
        #     target_heatmaps = ret['heatmaps'][0]
        #     img = ret['imgs'][0]
        #     orig_img = ret['orig_imgs'][0]
        #     visibility = ret['visibility'][0]
        #     for k in range(0,21,6):
            
        #         fig = plt.figure()
        #         ax1 = fig.add_subplot(131)
        #         ax2 = fig.add_subplot(132)
        #         ax3 = fig.add_subplot(133)
        #         print('subpixel:',pose2d[k])
        #         print('pixel:',np.argmax(target_heatmaps[k]) - np.argmax(target_heatmaps[k]) // 64 * 64, np.argmax(target_heatmaps[k]) // 64)
        #         ax1.imshow(cv2.cvtColor(np.transpose((img).numpy(),(1,2,0)), cv2.COLOR_RGB2BGR))
        #         ax3.imshow(orig_img)
        #         plot_hand(ax1, 4*pose2d[:,0:2], order='uv')
        #         plot_hand(ax2, pose2d[:,0:2], order='uv')
        #         ax2.imshow(target_heatmaps[k])
        #         plt.title('MHP: {} Joint id: {} Vis: {}'.format(idx, k, visibility[k]==1))
        #         plt.show()
        return ret

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {} ({} views: {})\n'.format(self.__len__(), len(self.n_views), str(self.n_views))
        fmt_str += '    Sample length: {}\n'.format(self.seq_len)
        fmt_str += '    Sample stride: {}\n'.format(self.stride)
        fmt_str += '    Adjacent frames: {}\n'.format(self.seq_idx)
        fmt_str += '    Root Location: {}\n'.format(self.data_dir)
       #fmt_str = fmt_str + 'Data directories: ' + str(self.data_list)
        return fmt_str
    
    def visualize_data(self):
        for i in range(self.__len__()):
            # img (torch.tensor): N_frames x 3 x H x W
            # pose2d_gts: N_frames x 21 x 3 [u,v,vis]
            # pose3d_gts: N_frames x 21 x 3 (cam coord)
            imgs, pose2d_gts, pose3d_gts = self.__getitem__(i)
            self._visualize_data(imgs, pose2d_gts, pose3d_gts)

    def _visualize_data(self, imgs, pose2d_gts, pose3d_gts):
        # img (torch.tensor): N_frames x 3 x H x W
        # target_heatmaps: N_frames x 21 x 64(H) x 64(W)
        # pose2d_gts: N_frames x 21 x 3 [u,v,vis]
        # pose3d_gts: N_frames x 21 x 3 (cam coord)

        im_height, im_width = imgs.shape[:2]

        for frame_id in range(imgs.shape[0]):
            img = cv2.cvtColor(np.transpose(imgs[frame_id],(1,2,0)), cv2.COLOR_RGB2BGR)
            pose2d = pose2d_gts[frame_id]
            pose3d = pose3d_gts[frame_id]
            #pose3d = pose3d - pose3d[0]
            print(pose3d)

            fig = plt.figure(frame_id)
            # fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

            # 1. plot raw image
            ax = fig.add_subplot(1,2,1)  
            ax.imshow(img)
            
            for i in range(21):
                if pose2d[i,2] == 0:
                    plt.scatter(pose2d[i,0],pose2d[i,1])
            visualize_joints_2d(ax, pose2d, joint_idxs=False)

            # 3. plot 3D joints
            ax = fig.add_subplot(122, projection='3d')
            draw_3d_skeleton_on_ax(pose3d, ax)
            ax.set_title("GT 3D joints")

            plt.show()
