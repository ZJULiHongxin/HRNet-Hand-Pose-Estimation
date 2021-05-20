# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# python infer_3D.py --cfg ../experiments\LearnableTriangulation/VolTriangulation_MHP_v2.yaml --model_path ../output\LearnableTriangulation\VolTriangulation_MHP_v2/model_best.pth.tar --gpu 0
# python infer_3D.py --cfg ../experiments\LearnableTriangulation/RANSACTriangulation_MHP_v1.yaml --gpu 0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings
import numpy as np
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
# plt.rcParams['savefig.dpi'] = 150 #图片像素
# plt.rcParams['figure.dpi'] = 150 #分辨率

import _init_paths
from config import cfg
from config import update_config

from fp16_utils.fp16util import network_to_half
from core.loss import Joints3DMSELoss, BoneLengthLoss, JointAngleLoss, JointsMSELoss

import dataset
from dataset.frei_utils.fh_utils import *
from dataset.HandGraph_utils.vis import *
from dataset import build_transforms

from models import pose_hrnet, pose_hrnet_softmax
from models.FTL_encoder_decoder import FTLMultiviewNet
from models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet, VolumetricTriangulationNet_CPM

from utils.heatmap_decoding import get_final_preds
from utils.transforms import scale_pose3d
from utils.misc import DLT, DLT_pytorch, DLT_sii_pytorch, update_after_resize
#from .HandGraph_utils.vis import *

def parse_args():
    parser = argparse.ArgumentParser(description='Please specify the mode [training/assessment/predicting]')
    parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
    parser.add_argument('opts',
                    help="Modify cfg options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        default=-1,
                        type=int)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--is_vis',
                    default=1,
                    type=int)
    parser.add_argument('--data_dir',
                    default='E:/Hand_Datasets',
                    type=str)
    parser.add_argument('--model_path', default='', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.freeze()

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_path = args.model_path
    is_vis = args.is_vis

    gpus = ','.join([str(i) for i in cfg.GPUS])
    gpu_ids = eval('['+gpus+']')

    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")
    
    # model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=True
    # )

    if 'pose_hrnet' in cfg.MODEL.NAME:
        model = {
            "pose_hrnet": pose_hrnet.get_pose_net,
            "pose_hrnet_softmax": pose_hrnet_softmax.get_pose_net
        }[cfg.MODEL.NAME](cfg, is_train=True)
    else:
        model = {
            "ransac": RANSACTriangulationNet,
            "alg": AlgebraicTriangulationNet,
            "vol": VolumetricTriangulationNet,
            "vol_CPM": VolumetricTriangulationNet_CPM,
            "FTL": FTLMultiviewNet
        }[cfg.MODEL.NAME](cfg, is_train=False)

    
    # load model state
    if model_path:
        print("Loading model:", model_path)
        ckpt = torch.load(model_path, map_location='cpu' if args.gpu == -1 else 'cuda:0')
        if 'state_dict' not in ckpt.keys():
            state_dict = ckpt
        else:
            state_dict = ckpt['state_dict']
            print('Model epoch {}'.format(ckpt['epoch']))
        
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        
        model.load_state_dict(state_dict, strict=True)
        
    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not args.distributed:
        print('Warning: Sync BatchNorm is only supported in distributed training.')

    device = torch.device('cuda:'+str(args.gpu) if args.gpu != -1 else 'cpu')
    
    model.to(device)
    
    model.eval()

    # image transformer
    transform = build_transforms(cfg, is_train=False)
    
    inference_dataset = eval('dataset.'+cfg.DATASET.TEST_DATASET[0])(
        cfg,
        cfg.DATASET.TEST_SET,
        transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    print('\nValidation loader information:\n' + str(data_loader.dataset))
    
    with torch.no_grad():
        pose2d_mse_loss = JointsMSELoss().to(device) if args.gpu != -1 else JointsMSELoss()
        pose3d_mse_loss = Joints3DMSELoss().to(device) if args.gpu != -1 else Joints3DMSELoss()
        orig_width, orig_height = inference_dataset.orig_img_size
        heatmap_size = cfg.MODEL.HEATMAP_SIZE
        count = 4
        for i, ret in enumerate(data_loader):
            # orig_imgs: 1 x 4 x 480 x 640 x 3
            # imgs: 1 x 4 x 3 x H x W
            # pose2d_gt (bounded in 64 x 64): 1 x 4 x 21 x 2 
            # pose3d_gt: 1 x 21 x 3
            # visibility: 1 x 4 x 21
            # extrinsic matrix: 1 x 4 x 3 x 4 
            # intrinsic matrix: 1 x 3 x 3
            if not (i % 67 == 0): continue

            imgs = ret['imgs'].to(device)
            orig_imgs = ret['orig_imgs']
            pose2d_gt, pose3d_gt, visibility = ret['pose2d'], ret['pose3d'], ret['visibility']
            extrinsic_matrices, intrinsic_matrices = ret['extrinsic_matrices'], ret['intrinsic_matrix']
            # somtimes intrisic_matrix has a shape of 3x3 or b x 3x3
            intrinsic_matrix = intrinsic_matrices[0] if len(intrinsic_matrices.shape) == 3 else intrinsic_matrices
            
            
            start_time = time.time()
            if 'pose_hrnet' in cfg.MODEL.NAME:
                pose3d_gt = pose3d_gt.to(device)

                heatmaps, _ = model(imgs[0]) # N_views x 21 x 64 x 64
                pose2d_pred = get_final_preds(heatmaps, cfg) # N_views x 21 x 2
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices).to(device) # b x v x 3 x 4

                # rescale to the original image before DLT
                pose2d_pred[:,:,0:1] *= orig_width/heatmap_size[0]
                pose2d_pred[:,:,1:2] *= orig_height/heatmap_size[0]
                # 3D world coordinate 1 x 21 x 3
                pose3d_pred = DLT_pytorch(pose2d_pred, proj_matrices.squeeze()).unsqueeze(0)

            elif 'alg' == cfg.MODEL.NAME or 'ransac' == cfg.MODEL.NAME:
                # the predicted 2D poses have been rescaled inside the triangulation model
                # pose2d_pred: 1 x N_views x 21 x 2
                # pose3d_pred: 1 x 21 x 3
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices) # b x v x 3 x 4

                pose3d_pred,\
                pose2d_pred,\
                heatmaps,\
                confidences_pred = model(imgs, proj_matrices.to(device))

            elif "vol" in cfg.MODEL.NAME:
                intrinsic_matrix = update_after_resize(
                    intrinsic_matrix,
                    (orig_height, orig_width),
                    tuple(heatmap_size))
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices).to(device) # b x v x 3 x 4

                # pose3d_pred (torch.tensor) b x 21 x 3
                # pose2d_pred (torch.tensor) b x v x 21 x 2 NOTE: the estimated 2D poses are located in the heatmap size 64(W) x 64(H)
                # heatmaps_pred (torch.tensor) b x v x 21 x 64 x 64
                # volumes_pred (torch.tensor)
                # confidences_pred (torch.tensor)
                # cuboids_pred (list)
                # coord_volumes_pred (torch.tensor)
                # base_points_pred (torch.tensor) b x v x 1 x 2
                if cfg.MODEL.BACKBONE_NAME == 'CPM_volumetric':
                    centermaps = ret['centermaps'].to(device)

                    pose3d_pred,\
                    pose2d_pred,\
                    heatmaps_pred,\
                    volumes_pred,\
                    confidences_pred,\
                    coord_volumes_pred,\
                    base_points_pred\
                        = model(imgs, centermaps, proj_matrices)
                else:
                    pose3d_pred,\
                    pose2d_pred,\
                    heatmaps,\
                    volumes_pred,\
                    confidences_pred,\
                    coord_volumes_pred,\
                    base_points_pred\
                        = model(imgs, proj_matrices)
                
                pose2d_pred[:,:,:,0:1] *= orig_width/heatmap_size[0]
                pose2d_pred[:,:,:,1:2] *= orig_height/heatmap_size[0]
            
            elif 'FTL' == cfg.MODEL.NAME:
                # pose2d_pred: 1 x 4 x 21 x 2
                # pose3d_pred: 1 x 21 x 3
                heatmaps, pose2d_pred, pose3d_pred = model(
                    imgs.to(device),
                    extrinsic_matrices.to(device), intrinsic_matrix.to(device))

                print(pose2d_pred)
                pose2d_pred = torch.cat((pose2d_pred[:,:,:,0:1]*640/64, pose2d_pred[:,:,:,1:2]*480/64), dim=-1)

            # N_views x 21 x 2
            end_time = time.time()
            print('3D pose inference time {:.1f} ms'.format(1000*(end_time-start_time)))
            pose3d_EPE = pose3d_mse_loss(pose3d_pred[:,1:], pose3d_gt[:,1:].to(device)).item()
            print('Pose3d MSE: {:.4f}\n'.format(pose3d_EPE))

            # if pose3d_EPE > 35:
            #     input()
            #     continue
            # 2D errors
            pose2d_gt[:,:,:,0] *= orig_width / heatmap_size[0]
            pose2d_gt[:,:,:,1] *= orig_height / heatmap_size[1]

            # for k in range(21):
            #     print(pose2d_gt[0,k].tolist(), pose2d_pred[0,k].tolist())
            # input()

            visualize(
                args=args, imgs=np.squeeze(orig_imgs[0].numpy()),
                pose2d_gt=np.squeeze(pose2d_gt.cpu().numpy()),
                pose2d_pred=np.squeeze(pose2d_pred.cpu().numpy()),
                pose3d_gt=np.squeeze(pose3d_gt.cpu().numpy()),
                pose3d_pred=np.squeeze(pose3d_pred.cpu().numpy())
            )
            
 


def visualize(args, imgs, pose2d_gt, pose2d_pred, pose3d_gt, pose3d_pred):
    # imgs: n_views x H x W x 3
    # pose2d_gt: v x 21 x 2
    # pose2d_pred: v x 21 x 2
    # pose3d_gt: 21 x 3
    # pose3d_pred: 21 x 3
    fig = plt.figure(1)
    fig.set_size_inches(float(4 * 256) / fig.dpi, float(4 * 256) / fig.dpi, forward=True)
    
    """
    compare
    """
    if 'ransac' in args.cfg.lower():
        ylabel = 'RANSAC'
    elif 'vol' in args.cfg.lower():
        ylabel = 'Volumetric'
    else:
        ylabel = 'DLT'
    fig0 = plt.figure(1)
    for v in range(imgs.shape[0]):
        ax1 = fig0.add_subplot(2,imgs.shape[0],v+1)
        #print(pose2d_pred[v].numpy())
        img_np = imgs[v]
        ax1.imshow(img_np)
        plot_hand(ax1, pose2d_gt[v], order='uv', draw_kp=False)
        ax1.set_title('View {}'.format(v+1))
        if v == 0: 
            ax1.set_ylabel('Groundtruth')
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig0.add_subplot(2,imgs.shape[0],v+1+imgs.shape[0])
        ax2.imshow(img_np)
        if v == 0: 
            ax2.set_ylabel(ylabel)
            
        ax2.set_xticks([])
        ax2.set_yticks([])
        plot_hand(ax2, pose2d_pred[v], order='uv', draw_kp=False)

    fig = plt.figure(2)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    draw_3d_skeleton_on_ax(pose3d_gt, ax1)
    ax1.set_title('Groundtruth')
    ax1.view_init(elev=30., azim=100)

    draw_3d_skeleton_on_ax(pose3d_pred, ax2)
    ax2.set_title('rec')
    ax2.view_init(elev=30., azim=100)
    # plt.xlim(-250,250)
    # plt.ylim(-250,250)
    plt.show()


    if args.is_vis:
        plt.show()
    else:
        ret = fig2data(fig)
        plt.close(fig)
        import os
        result_dir = './Pred_vis'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        cv2.imwrite(os.path.join(result_dir,'{:06d}.jpg'.format(i)), ret)
        print('write ',os.path.join(result_dir,'{:06d}.jpg'.format(i)))
    
main()