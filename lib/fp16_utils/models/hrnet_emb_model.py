# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from .pose_hrnet import get_pose_net
from .pose_hrnet_trainable_softmax import get_pose_net as get_pose_net_trainable_softmax

class TemporalModel(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, in_channels, num_joints_out, filter_widths, causal=False, dropout=0.5, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__()
        
        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.in_channels = in_channels
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)
        self.expand_conv = nn.Conv1d(in_channels, channels, filter_widths[0], bias=False) # no padding

        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
    
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames
    
    def forward(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x)))) # nn.Dropout(dropout)

        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            # The length of each channel will be reduced by the value of dilation after a residual block
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))

        x = self.shrink(x)

        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.residual_conv = conv3x3(inplanes, planes, stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.residual_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        channel_0 = config.DATASET.NUM_JOINTS + 2*config.MODEL.EXTRA.STAGE4.NUM_CHANNELS[0] # 21 + 32
        channel_1 = 128
        channel_2 = 256
        channel_3 = 512
        channel_lst = [channel_0, 128]
        self.block = []
        for i in range(len(channel_lst)-1):
            self.block.append(BasicBlock(channel_lst[i], channel_lst[i+1])) # -> b x 64 x 64 x 64 
        self.block = torch.nn.ModuleList(self.block)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.block2 = BasicBlock(channel_1, channel_2)
        #self.block3 = BasicBlock(channel_2, channel_3)
        in_channel = channel_lst[-1] * (config.MODEL.HEATMAP_SIZE[0] // 2**(len(channel_lst)-1))**2
        self.fc1 = nn.Linear(in_channel, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, config.MODEL.EMBEDDING_SIZE)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # img: a tensor  generated by concatenating intermidiate features, heatmap and the last frame
        # size: b x N_FrAMES x 64 x 64
        for i in range(len(self.block)):
            x = self.downsample(self.block[i](x))  # -> b x 64 x 32 x 32
        #x = self.downsample(self.block2(x)) # -> b x 128 x 16  x 16
        #x = self.downsample(self.block3(x)) # -> b x 256 x 8 x 8
        x_flattened = x.reshape(x.shape[0], -1)
        x_linear = self.relu1(self.fc1(x_flattened))
        x_linear = self.relu2(self.fc2(x_linear))

        return x_linear # size b x emb_size


class  HRNet_Emb_TCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # split input data into small chunks due to shortage of GPU RAM
        self.split_size = 4
        
        # Pretrained HRNet
        if config.MODEL.NAME == 'pose_hrnet':
            self.HRNet = get_pose_net(config, is_train=True)
        if config.MODEL.NAME == 'pose_hrnet_trainable_softmax':
            self.HRNet = get_pose_net_trainable_softmax(config, is_train=True)
        
        if config.MODEL.HRNET_PRETRAINED:
            checkpoint = torch.load(config.MODEL.HRNET_PRETRAINED, map_location='cpu')
            print("=> loading a pretrained HRNet model '{}' (Epoch: {})".format(config.MODEL.HRNET_PRETRAINED, checkpoint['epoch']))
            self.HRNet.load_state_dict(checkpoint['state_dict'], strict=False)
        
            for p in self.HRNet.parameters():
                p.requires_grad = False
        
        # embeding net
        self.emb_net = ResNet(config)
        
        # TCN
        filter_widths = config.MODEL.FILTER_WIDTHS
        self.TCN = TemporalModel(
            in_channels=config.MODEL.EMBEDDING_SIZE,
            num_joints_out=config.DATASET.NUM_JOINTS,
            filter_widths=filter_widths,
            causal=0,
            dropout=0.5,
            channels=config.MODEL.TCN_CHANNELS)
        

    def forward(self, frames):
        # frames: b x N_frames x 3 x H x W
        emb_lst = []

        for b in range(frames.shape[0]):
            heatmaps, (softmax_temp, inter_feat) = self.HRNet(frames[b]) # b x 21 x H x W

            embeddings = self.emb_net(torch.cat((heatmaps, inter_feat), dim=1)) # b x emb_size
            
            emb_lst.append(embeddings)

        emb = torch.stack(emb_lst) # b x N_frames x emb_size
        print('TCN',emb.shape)
        x = self.TCN(torch.transpose(emb, 1, 2)) # b x N_frames x emb_size -> b x 63 x reduced_length
        x_middle = x[:,:,x.shape[2]//2]

        return x_middle.reshape((x.shape[0], -1,3))
