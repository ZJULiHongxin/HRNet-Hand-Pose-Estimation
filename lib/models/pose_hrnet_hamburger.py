import math
import os.path as osp
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


from .hamburger import ConvBNReLU, get_hamburger
from . import pose_hrnet_softmax, pose_resnet

# from sync_bn.nn.modules import SynchronizedBatchNorm2d

class HamNet(nn.Module):
    def __init__(self, config, is_train):
        super().__init__()
        self.config = config
        # HRNet backbone
        input_size = config.MODEL.HEATMAP_SIZE[0]
        
        in_channel = 3
        if 'hrnet' in config.MODEL.BACKBONE_NAME: # Use HRNet backbone
            in_channel = sum(config.MODEL.EXTRA.STAGE4.NUM_CHANNELS)
            self.backbone = eval(config.MODEL.BACKBONE_NAME + '.get_pose_net(config, is_train=True)')
            backbone_path = config.MODEL.BACKBONE_MODEL_PATH
            if backbone_path and is_train:
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
                
            # freeze lower layers
        elif 'resnet' in config.MODEL.BACKBONE_NAME:
            in_channel = 480
            self.backbone = eval(config.MODEL.BACKBONE_NAME + '.get_pose_net(config, is_train=True)')
        
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Hamburger
        embed_dim = config.MODEL.EMB_DIM
        self.squeeze = ConvBNReLU(in_channel, embed_dim, 3)

        Hamburger = get_hamburger(config.MODEL.VERSION)
        self.hamburger = Hamburger(config, in_c=embed_dim)
        
        self.align = ConvBNReLU(embed_dim, 256, 3)
        self.fc = nn.Sequential(nn.Dropout2d(p=0.1),
                                nn.Conv2d(256, config.DATASET.NUM_JOINTS, 1))
        self.trainable_temp = torch.nn.parameter.Parameter(torch.tensor(1.0), requires_grad=config.MODEL.TRAINABLE_SOFTMAX)

    def forward(self, x):
        # x: b x 3 x H x W
        if 'hrnet' in self.config.MODEL.BACKBONE_NAME:
            _, x, _ = self.backbone(x) # b x 21 x 64 x 64 else b x 3 x 256 x 256
        elif 'resnet' in self.config.MODEL.BACKBONE_NAME:
            x = self.backbone(x)
        print(x.shape)
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        x = self.fc(x)

        # Apply 2D softmax to generate heatmaps
        x_flattened = x.view(x.shape[0], x.shape[1], -1)
        x_softmax = F.softmax(x_flattened * self.trainable_temp, dim=2)

        heatmap_pred = x_softmax.view(x.shape)

        return heatmap_pred, self.trainable_temp

def get_pose_net(cfg, is_train, **kwargs):
    model = HamNet(cfg, is_train, **kwargs)

    return model