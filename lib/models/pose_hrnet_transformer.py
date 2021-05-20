## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# <3D Human Pose Estimation with Spatial and Temporal Transformers>
import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from models import pose_hrnet_softmax
from utils.heatmap_decoding import get_final_preds

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PoseTransformer(nn.Module):
    def __init__(self, config, is_train):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        num_frame = len(config.DATASET.SEQ_IDX)
        num_joints = config.DATASET.NUM_JOINTS
        in_chans=2
        embed_dim_ratio=32
        depth=4    
        num_heads=8
        mlp_ratio=2.
        qkv_bias=True
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0.2
        norm_layer=None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 2     #### output dimension is num_joints * 3

        ### pretrained HRNet
        self.backbone = eval(config.MODEL.BACKBONE_NAME + '.get_pose_net(config, is_train=True)')
        #self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        if is_train:
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
                
            # freeze lower layers
            if 'hrnet' in config.MODEL.BACKBONE_NAME:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for p in self.backbone.stage3.parameters():
                    p.requires_grad = False
                for p in self.backbone.stage4.parameters():
                    p.requires_grad = True
                for p in self.backbone.last_layer.parameters():
                    p.requires_grad = True
                try:
                    self.backbone.trainable_temp.requires_grad = False
                except:
                    print('No trainable temperature')
                    pass
        self.use_softmax = config.MODEL.HEATMAP_SOFTMAX
        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b  = x.shape[0]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x


    def forward(self, x):
        # x: b x f x 3 x H x W
        n_batches, n_frames = x.shape[0:2]
        heatmaps_pred, trainable_temp = self.backbone(x.view(-1, *x.shape[-3:])) # b*f x 21 x 64 x 64
        n_joints = heatmaps_pred.shape[1]
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=self.use_softmax).view(n_batches, n_frames, n_joints, 2) # b x f x 21 x 2

        x = pose2d_pred.permute(0, 3, 1, 2)
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x) # b x N_frames x (N_joints * 32)
        x = self.forward_features(x) # b x 1 x (N_joints * 32)
        pose2d_pred_refined = self.head(x) # b x 1 x (N_joints * 2)

        return pose2d_pred_refined.view(n_batches, n_joints, -1), heatmaps_pred, trainable_temp


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseTransformer(cfg, is_train, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model