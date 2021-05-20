# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from . import pose_hrnet_softmax, pose_hrnet_volumetric, pose_hrnet

class ChannelWiseFC(nn.Module):

    def __init__(self, size):
        super(ChannelWiseFC, self).__init__()
        self.weight = nn.Linear(size, size, bias=False)
        #self.weight = nn.Parameter(torch.Tensor(size, size))
        #self.weight.data.uniform_(0, 0.1)

    def forward(self, input):
        N, C, H, W = input.size()
        input_reshape = input.reshape(N * C, H * W)
        #output = torch.matmul(input_reshape, self.weight)
        output = self.weight(input_reshape)
        output_reshape = output.reshape(N, C, H, W)
        return output_reshape


class Aggregation(nn.Module):

    def __init__(self, cfg, weights=[0.4, 0.2, 0.2, 0.2]):
        super(Aggregation, self).__init__()
        NUM_NETS = 4 * (4-1) # MHP has 4 iews
        size = cfg.MODEL.HEATMAP_SIZE[0]
        self.weights = weights
        self.aggre = nn.ModuleList()
        for i in range(NUM_NETS):
            self.aggre.append(ChannelWiseFC(size * size))

    def sort_views(self, target, all_views):
        indicator = [target is item for item in all_views]
        new_views = [target.clone()]
        for i, item in zip(indicator, all_views):
            if not i:
                new_views.append(item.clone())
        return new_views

    def fuse_with_weights(self, views):
        target = torch.zeros_like(views[0])
        for v, w in zip(views, self.weights):
            target += v * w
        return target

    def forward(self, inputs):
        index = 0
        outputs = []
        nviews = len(inputs)
        for i in range(nviews):
            sorted_inputs = self.sort_views(inputs[i], inputs)
            warped = [sorted_inputs[0]]
            for j in range(1, nviews):
                fc = self.aggre[index]
                fc_output = fc(sorted_inputs[j])
                warped.append(fc_output)
                index += 1
            output = self.fuse_with_weights(warped)
            outputs.append(output)
        return outputs


class MultiViewPoseNet(nn.Module):

    def __init__(self, config):
        super(MultiViewPoseNet, self).__init__()
        self.config = config

        # load a backbone model
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

        # freeze lower layers
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.stage4.parameters():
            p.requires_grad = True
        for p in self.backbone.last_layer.parameters():
            p.requires_grad = True

        # cross-view fusion layer
        self.aggre_layer = Aggregation(config)

    def forward(self, views):
        # views: b x v x 3 x H x W
        if len(views.shape) == 4:
            views = torch.unsqueeze(views, dim=0)
        n_views = views.shape[1]
        single_views = []
        for view in range(n_views):
            heatmaps, _ = self.backbone(views[:,view]) # b x 21 x 64 x 64
            single_views.append(heatmaps)
        multi_views = []
        if self.config.MODEL.AGGRE:
            multi_views = self.aggre_layer(single_views) # a list of b x 21 x 64 x 64
            return torch.cat(multi_views, dim=0), torch.cat(single_views, dim=0)
        else:
            return torch.cat(single_views, dim=0)

def get_pose_net(cfg, is_train=None, **kwargs):
    model = MultiViewPoseNet(cfg, **kwargs)
    return model
