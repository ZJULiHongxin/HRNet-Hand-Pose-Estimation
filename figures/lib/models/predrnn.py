__author__ = 'yunbo'

import torch
import torch.nn as nn
from .pose_hrnet import get_pose_net

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 7, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t) # The output of Conv2D is a 4D tensor (size: [Batchsize, Channel, Height, Width])
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        # Split the processed input along the second dimension (Channel)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1) 
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()

        self.config = config
        self.frame_channel = config.MODEL.EXTRA.STAGE2.NUM_CHANNELS[0] + 21
        self.num_hidden = config.MODEL.N_HIDDEN
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = config.MODEL.HEATMAP_SIZE[0]

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], width, config.MODEL.FILTER_SIZE,
                                       config.MODEL.STRIDE, config.MODEL.LAYER_NORM)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames):
        # [batch, length, channel, height, width]
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        # initialize all the hidden states (short-term memory) and cell states (long-term memory)
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width])
            if frames.is_cuda:
                zeros = zeros.cuda()
                
            h_t.append(zeros)
            c_t.append(zeros)

        # The proposed temporal-spatial memory is updated in a zig-zag manner
        memory = torch.zeros([batch, self.num_hidden[0], height, width])
        if frames.is_cuda:
            memory = memory.cuda()
            
        for t in range(frames.shape[1]): # for every time step
            net = frames[:,t]
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers): # for every layer (upwards)
                
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)  # The non-first layers take as input the hidden states of the previous layer

            x_gen = self.conv_last(h_t[self.num_layers-1]) # predict one frame
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] - > [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()

        return next_frames

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
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

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
    def __init__(self, config, in_channel):
        super(ResNet, self).__init__()
        self.out_ch = in_channel
        self.block1 = BasicBlock(in_channel, self.out_ch)
        self.block2 = BasicBlock(self.out_ch, self.out_ch)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(self.out_ch \
            * config.MODEL.HEATMAP_SIZE[0] * config.MODEL.HEATMAP_SIZE[0] // 4, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 63)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, feat_img):
        # feat_img: a tensor  generated by concatenating intermidiate features, heatmap and the last frame
        # size: b x (32+21) x 64 x 64
        out = self.block1(feat_img)
        out = self.downsample(out)
        out = self.block2(out)

        out_flattened = out.reshape(out.shape[0], -1)
        out_linear = self.relu1(self.fc1(out_flattened))
        out_linear = self.relu2(self.fc2(out_linear))

        return out_linear # size b x 63

class HRNet_PredRNN_ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.RNN_in_channel = config.MODEL.EXTRA.STAGE2.NUM_CHANNELS[0] + 21
        self.RNN_feat_size = self.config.MODEL.HEATMAP_SIZE[0]

        """
        Three models
        # model 1: HRNet generates heatmaps and feature maps (10.1 GFLOPS)
        # model 2: PredRNN processes N successive frames (24.66 GFLOPS)
        # model 3: resnet outputs 3D pose predictions (0.29 GFLOPS)
        """
        self.HRNet = get_pose_net(config, is_train=True)
        if config.MODEL.HRNET_PRETRAINED:
            checkpoint = torch.load(config.MODEL.HRNET_PRETRAINED)
            print("=> loading a pretrained HRNet model '{}' (Epoch: {})".format(config.MODEL.HRNET_PRETRAINED, checkpoint['epoch']))
            self.HRNet.load_state_dict(checkpoint['state_dict'], strict=False)

        self.PredRNN = RNN(config)
        self.resnet = ResNet(config, self.RNN_in_channel)

    def forward(self, frames):
        # frames: B x L x 3 x H x W
        feat_heatmap = torch.zeros(
            size=(frames.shape[0], frames.shape[1], self.RNN_in_channel, self.RNN_feat_size, self.RNN_feat_size),
            dtype=torch.float32)
        
        # 1. extract feature maps and heat maps
        with torch.no_grad():
            for b in range(frames.shape[0]):
                # heatmap: b x 21 x H x W
                # feat: b x 64 x H x W
                heatmap, feat = self.HRNet(frames[b])
                feat_heatmap[b] = torch.cat((heatmap, feat), dim=1) # len x (32+32+21) x 64 x 64


        if frames.is_cuda:
            feat_heatmap = feat_heatmap.cuda()

        # 2. process all frames
        out_RNN = self.PredRNN(feat_heatmap) # batch x len x (32+21) x H x W
        
        out_RNN_last_frame = torch.squeeze(out_RNN[:,-1,:,:,:], dim=1) # batch x (32+21) x H x W
        # 3. output predicted pose
        out_ResNet = self.resnet(out_RNN_last_frame)
        pose3d_pred = out_ResNet.view((-1, 21, 3))

        return pose3d_pred


