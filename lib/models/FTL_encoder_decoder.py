import torch
from torch import nn
import torch.nn.functional as F

import utils
from utils.heatmap_decoding import get_final_preds
from utils.misc import DLT_sii_pytorch
from . import pose_hrnet_softmax, pose_hrnet_volumetric, pose_hrnet

BN_MOMENTUM = 0.1

class conv_block(nn.Module):
    def __init__(self, n_layers, channel_lst, kernel_size_lst, stride_lst, padding_lst):
        super().__init__()
        assert n_layers == len(channel_lst) - 1 \
            and n_layers == len(kernel_size_lst)\
            and len(kernel_size_lst) == len(stride_lst)\
            and len(stride_lst) == len(padding_lst)\
                , 'inconsistent number of hyper-parameters'
        
        layer_lst = []
        for i in range(n_layers):
            layer_lst.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channel_lst[i],
                        out_channels=channel_lst[i+1],
                        kernel_size=kernel_size_lst[i],
                        stride=stride_lst[i],
                        padding=padding_lst[i]),
                        nn.BatchNorm2d(channel_lst[i+1], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)
                )   
            )
        
        self.layer_lst = nn.ModuleList(layer_lst)
    
    def forward(self, x):
        for layer in self.layer_lst:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, channel_lst, kernel_size_lst, stride_lst, padding_lst, output_padding_lst):
        """
        (n_layers-1) deconvolution layers and 1 conv layer
        """
        super().__init__()
        assert n_layers == len(channel_lst) - 1 \
            and n_layers == len(kernel_size_lst)\
            and len(kernel_size_lst) == len(stride_lst)\
            and len(stride_lst) == len(padding_lst)\
            and len(padding_lst) == len(output_padding_lst) + 1\
                , 'inconsistent number of hyper-parameters'

        layer_lst = []
        for i in range(n_layers-1):
            layer_lst.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                    in_channels=channel_lst[i], out_channels=channel_lst[i+1],
                    kernel_size=kernel_size_lst[i], stride=stride_lst[i], padding=padding_lst[i],
                    output_padding=output_padding_lst[i]),
                    nn.BatchNorm2d(channel_lst[i+1], momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=False)
                )
            )
        
        layer_lst.append(
            nn.Conv2d(
                in_channels=channel_lst[-2], out_channels=channel_lst[-1],
                kernel_size=kernel_size_lst[-1], stride=stride_lst[-1],
                padding=padding_lst[-1])
        )

        self.layer_lst = nn.ModuleList(layer_lst)
    
    def forward(self, x):
        for layer in self.layer_lst:
            x = layer(x)
        return x

class FTLMultiviewNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()
        self.config = config
        # 1. 编码器：HRNet_w48+final_layer，输出bx(32+64+128+256=480)x18x18
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

        for p in self.backbone.parameters():
            p.requires_grad = False

        feat_map_channels = sum(config.MODEL.EXTRA.STAGE4.NUM_CHANNELS) # 32 + 64 + 128 + 256 = 480
        self.encoder_head = conv_block(
            n_layers=2, channel_lst=[feat_map_channels, feat_map_channels, 240],
            kernel_size_lst=[3,3], stride_lst=[2,2], padding_lst=[2,2])

        # 7. 2层1x1卷积: bx240x16x16
        compressed_feat_channels = feat_map_channels // 2
        self.fuse_after_FTL = conv_block(
            n_layers=2, channel_lst=[compressed_feat_channels * config.DATASET.NUM_VIEWS, compressed_feat_channels, compressed_feat_channels],
            kernel_size_lst=[1,1], stride_lst=[1,1], padding_lst=[0,0]
        )
        
        # 9. 1x1卷积：bx480x16x16
        self.channel_expansion = conv_block(
            n_layers=1, channel_lst=[compressed_feat_channels, feat_map_channels],
            kernel_size_lst=[1], stride_lst=[1], padding_lst=[0]
        )

        # 10.   转置卷积nn.ConvTranspose2d(16, 16, 3, stride=1,paddding=1)：bx256x16x16
        #       转置卷积nn.ConvTranspose2d(16, 16, 3, stride=2)：bx256x33x33
        #       转置卷积nn.ConvTranspose2d(16, 16, 3, stride=2,padding=1)：bx256x66x66
        #       3x3卷积层: bx21x64x64
        self.decoder = Decoder( 
            n_layers=3, channel_lst=[feat_map_channels, 256, 256, 256],
            kernel_size_lst=[3,3,3], stride_lst=[2,2,1], padding_lst=[2,2,1], output_padding_lst=[0,1] 
            )

        # 11.   1x1卷积层输出热力图
        self.final_layer = nn.Conv2d(
            in_channels=256, out_channels=config.DATASET.NUM_JOINTS,
            kernel_size=1, stride=1, padding=0
        )

    def forward(self, images, extrinsic_matrices=torch.rand((1,4,3,4)), intrinsic_matrices=torch.rand((1,3,3))):
        # images: b x v x 3 x H x W
        # extrinsic_mat (w2c): b x v x 3 x 4
        # intrinsic_mat (replicas of the same matreix): b x 3 x 3
        device = images.device
        batch_size, n_views = images.shape[:2]
        intrinsic_matrix = intrinsic_matrices[0]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # 编码器：HRNet_w48+final_layer，输出bx(32+64+128+256=480)x18x18
        heatmaps, inter_feat = self.backbone(images) # inter_feat: b*v x 480 x 64 x 64
        feature_maps = self.encoder_head(inter_feat) # inter_feat: b*v x 240 x 18 x 18

        # 2. 重塑： (the last dimension stands for a homogeneous 2D image coord)
        reshaped_features = feature_maps.view(batch_size, n_views, feature_maps.shape[1], -1, 3) # b x v x 240 x 108 x 3

        # 3. FTL
        R_T = extrinsic_matrices[:,:,:,0:-1].transpose(2,3) # b x v x 3 x 3
        t_T = extrinsic_matrices[:,:,:,-1:].transpose(2,3) # b x v x 1 x 3
        intrinsic_matrix_T = intrinsic_matrix.T # 3 x 3

        canonical_feature_lst = []
        for v in range(n_views):
            # pose2d -> 3D cam coord
            canonical_feature = torch.matmul(reshaped_features[:,v], torch.inverse(intrinsic_matrix_T)) # b x 240 x 108 x 3
            # 3D cam coord -> 3D world coord.
            canonical_feature = torch.matmul(canonical_feature - t_T[:,v:v+1], torch.inverse(R_T[:,v:v+1]))  # b x 240 x 108 x 3
            # 4. 重塑
            canonical_feature_lst.append(canonical_feature.view(batch_size, *feature_maps.shape[1:])) # b x 240 x 18 x 18

        # 6. 合并张量：v个视角，合并得到bx240vx16x16
        canonical_feature_all_views = torch.cat(canonical_feature_lst, dim=1) # b x 240*v x 18 x 18

        # 7. 2层1x1卷积
        fused_features = self.fuse_after_FTL(canonical_feature_all_views).view(batch_size, *reshaped_features.shape[2:]) # b x 240 x 108 x 3

        # 8. FTL分发：bx240x16x16
        features_each_view_lst = []
        for v in range(n_views):
            features_each_view = torch.matmul(fused_features, R_T[:,v:v+1]) + t_T[:,v:v+1] # b x 240 x 108 x 3
            features_each_view = torch.matmul(features_each_view, intrinsic_matrix_T) # b x 240 x 108 x 3
            features_each_view_lst.append(features_each_view.view(batch_size, *feature_maps.shape[1:])) # b x 240 x 18 x 18

        features_all_views = torch.cat(features_each_view_lst, dim=0) # b*v x 240 x 18 x 18

        # 9. 1x1卷积
        expanded_features = self.channel_expansion(features_all_views) # b*v x 480 x 18 x 18

        # 10.   Decoder:
        #       转置卷积nn.ConvTranspose2d(16, 16, 3, stride=1,paddding=1)：b*v x 480 x 18 x 18
        #       转置卷积nn.ConvTranspose2d(16, 16, 3, stride=2)：b*v x 480 x 18 x 18
        #       转置卷积nn.ConvTranspose2d(16, 16, 3, stride=2,padding=1)：b*v x 480 x 18 x 18
        #       3x3卷积层: bx21x64x64
        decoded_features = self.decoder(expanded_features) # b*v x 480 x 64 x 64

        # 11.   1x1卷积层输出热力图
        heatmaps = self.final_layer(decoded_features) # b*v x n_joints x 64 x 64

        # Apply 2D softmax to generate heatmaps
        heatmaps_flattened = heatmaps.view(heatmaps.shape[0], heatmaps.shape[1], -1) # b*v x n_joints x 64*64
        heatmaps_softmax = F.softmax(heatmaps_flattened, dim=2)
        heatmaps_pred = heatmaps_softmax.view(heatmaps.shape) # b*v x n_joints x 64 x 64
        
        pose2d_pred = get_final_preds(heatmaps_pred, use_softmax=True).view(batch_size, n_views, -1, 2) # b x v x n_joints x 2

        proj_matrices = torch.matmul(intrinsic_matrix, extrinsic_matrices) # b x v x 3 x 4

        pose3d_pred = torch.cat([DLT_sii_pytorch(proj_matrices, pose2d_pred[:,:,k]).unsqueeze(1) for k in range(pose2d_pred.shape[2])], dim=1)

        return heatmaps_pred, pose2d_pred, pose3d_pred