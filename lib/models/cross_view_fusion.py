from copy import deepcopy
import sys
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

import models
from utils.heatmap_decoding import get_final_preds
from utils.misc import DLT_pytorch, DLT, DLT_sii_pytorch
from utils.hand_skeleton import Hand
from .triangulation_model_utils import  op, multiview, volumetric
from . import pose_hrnet_softmax, pose_hrnet_volumetric, pose_hrnet

class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hand = Hand()
        self.num_joints = config.DATASET.NUM_JOINTS

        # a backbone used to estimate heatmaps
        self.backbone = eval(config.MODEL.BACKBONE_NAME + '.get_pose_net(config, is_train=True)')
        backbone_path = config.MODEL.BACKBONE_MODEL_PATH
        if backbone_path:
            checkpoint = torch.load(backbone_path, map_location='cpu')
            if 'state_dict' in checkpoint.keys():
                state_dict = checkpoint['state_dict']
                print("=> Loading pretrained {} backbone from '{}' (epoch {})".format(
                    config.MODEL.BACKBONE_NAME, backbone_path, checkpoint['epoch']))
            else:
                state_dict = checkpoint
                print("=> Loading pretrained {} backbone from '{}'".format(
                    config.MODEL.BACKBONE_NAME, backbone_path))

            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)
            
            self.backbone.load_state_dict(state_dict, strict=False)
            
        for p in self.backbone.last_layer.parameters():
            p.requires_grad = False


    def forward(self, images, proj_matrices, batch=None, keypoints_3d=None):
        # images: b x N_views x 3 x H x W
        # proj_matricies_batch (K*H): b x N_views x 3 x 4
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        # heatmaps: (b*N_views) x 21 x 64 x 64
        # features: (b*N_views) x 480 x 64 x 64
        heatmaps, _ = self.backbone(images)

        # find the middle finger root position
        base_idx = 9
        base_points_2d = get_final_preds(heatmaps, use_softmax=self.heatmap_softmax).view(batch_size, n_views, heatmaps.shape[1], 2) # batch_size x N_views x 21 x 2
        base_points = DLT_sii_pytorch(base_points_2d[:,:,9], proj_matrices[b]) # b x 3
        
        for b in range(batch_size):
            grid_center = base_points[b]
            limb_length = self.hand.compute_limb_length()
