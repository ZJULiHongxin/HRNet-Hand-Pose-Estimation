from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import platform
import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn

import _init_paths
from config import cfg
from config import update_config

from utils.utils import get_model_summary
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
from utils.misc import DLT_pytorch, DLT, DLT_sii_pytorch, update_after_resize
# python evaluate_3D.py --cfg ../experiments/JointTraining/JointTraining_v1.yaml --model_path ../output/JointTraining/JointTraining_v1/model_best.pth.tar --gpu 1
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
                    default=0,
                    type=int)
    parser.add_argument('--views',
                    default='[1,2,3,4]',
                    type=str)
    parser.add_argument('--batch_size',
                    default=32,
                    type=int)                 
    parser.add_argument('--model_path', default='', type=str)

    args = parser.parse_args()

    return args


def plot_performance(PCK2d, th2d_lst, PCK3d, th3d_lst, mse2d_each_joint, mse3d_each_joint):
    legend_lst = [
    # 0           1               2                  3                4
    'wrist', 'thumb palm', 'thumb near palm', 'thumb near tip', 'thumb tip',
    # 5                    6                 7                8
    'index palm', 'index near palm', 'index near tip', 'index tip',
    # 9                    10                  11               12
    'middle palm', 'middle near palm', 'middle near tip', 'middle tip',
    # 13                  14               15            16
    'ring palm', 'ring near palm', 'ring near tip', 'ring tip',
    # 17                  18               19              20
    'pinky palm', 'pinky near palm', 'pinky near tip', 'pinky tip', 'Avg']

    color = ['grey','gold','darkviolet','turquoise','r','g','b', 'c', 'm', 'y',
            'k','darkorange','lightgreen','plum', 'tan',
            'khaki', 'pink', 'skyblue','lawngreen','salmon','coral','maroon']

    # 2D pose mse
    plt.figure(1)
    plt.subplots_adjust(top=0.97, bottom=0.32, left=0.11, right=0.96, hspace=0.2, wspace=0.2)
    X = list(range(0,44,2))
    Y = np.concatenate((mse2d_each_joint, [mse2d_each_joint.mean()]))
    plt.bar(X, Y, width = 1.5, color = color)
    
    plt.xticks(X, legend_lst, rotation=270)
    plt.xlabel('Key Point')
    plt.ylabel('MSE [px]')
    plt.title('2D pose MSE. Average: {:.2f}'.format(mse2d_each_joint.mean()))
    print('2D pose EPE: {:.4f} px'.format(mse2d_each_joint.mean()))
    for x,y in zip(X,Y):
        plt.text(x+0.005,y+0.005,'%.2f' % y, fontsize=6, ha='center',va='bottom')

    # 3D pose mse
    plt.figure(2)
    plt.subplots_adjust(top=0.97, bottom=0.32, left=0.11, right=0.96, hspace=0.2, wspace=0.2)
    X = list(range(0,44,2))
    Y = np.concatenate((mse3d_each_joint, [mse3d_each_joint.mean()]))
    plt.bar(X, Y, width = 1.5, color = color)
    
    plt.xticks(X, legend_lst, rotation=270)
    plt.xlabel('Key Point')
    plt.ylabel('MSE [px]')
    plt.title('3D pose MSE. Average: {:.2f}'.format(mse3d_each_joint.mean()))
    print('3D pose EPE: {:.4f} mm'.format(mse3d_each_joint.mean()))
    for x,y in zip(X,Y):
        plt.text(x+0.005,y+0.005,'%.2f' % y, fontsize=6, ha='center',va='bottom')

    # 2/3D pose PCK
    start,end = 0, len(PCK2d)
    th2d_lst, PCK2d = th2d_lst[start:end], PCK2d[start:end]
    fig = plt.figure(3)
    plt.subplot(1,2,1)
    plt.plot(th2d_lst, PCK2d, marker='.')
    plt.xlabel('threshold [px]')
    plt.ylabel('PCK')
    # Area under the curve
    area = (PCK2d[0] + 2 * PCK2d[1:-1].sum() + PCK2d[-1])  * (th2d_lst[1] - th2d_lst[0]) / 2 / (th2d_lst[-1] - th2d_lst[0])
    plt.title('2D PCK AUC over all joints: {:.4f}'.format(area))
    print('2D PCKAUC: {:.4f}'.format(area))

    plt.subplot(1,2,2)
    start,end = 0, len(PCK2d)
    th3d_lst, PCK3d = th3d_lst[start:end], PCK3d[start:end]
    plt.plot(th3d_lst, PCK3d, marker='.')
    plt.ylim([0,1.0])
    plt.xlabel('threshold [mm]')
    plt.ylabel('PCK')

    # Area under the curve
    area = (PCK3d[0] + 2 * PCK3d[1:-1].sum() + PCK3d[-1])  * (th3d_lst[1] - th3d_lst[0]) / 2 / (th3d_lst[-1] - th3d_lst[0])
    plt.title('3D PCK AUC over all joints: {:.4f}'.format(area))
    print('3D PCKAUC: {:.4f}'.format(area))
    plt.tight_layout()
    plt.show()

result = 'eval3D_results'
prefix = './'+result+'/eval3D_results_'

def main():
    args = parse_args()


    update_config(cfg, args)
    cfg.defrost()
    cfg.freeze()

    if args.is_vis:
        result_dir = prefix+cfg.EXP_NAME
        mse2d_lst = np.loadtxt(os.path.join(result_dir, 'mse2d_each_joint.txt'))
        mse3d_lst = np.loadtxt(os.path.join(result_dir, 'mse3d_each_joint.txt'))
        PCK2d_lst = np.loadtxt(os.path.join(result_dir, 'PCK2d.txt'))
        PCK3d_lst = np.loadtxt(os.path.join(result_dir, 'PCK3d.txt'))

        plot_performance(PCK2d_lst[1,:], PCK2d_lst[0,:], PCK3d_lst[1,:], PCK3d_lst[0,:], mse2d_lst, mse3d_lst)
        exit()

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
    
    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not args.distributed:
        print('Warning: Sync BatchNorm is only supported in distributed training.')

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
        
        model.load_state_dict(state_dict, strict=False)
    
    device = torch.device('cuda:'+str(args.gpu) if args.gpu != -1 else 'cpu')
    
    model.to(device)
    
    model.eval()

    # image transformer
    transform = build_transforms(cfg, is_train=False)

    inference_dataset = eval('dataset.'+cfg.DATASET.DATASET[0])(
        cfg,
        cfg.DATASET.TEST_SET,
        transform=transform
    )
    inference_dataset.n_views = eval(args.views)
    batch_size = args.batch_size
    if platform.system()=='Linux': # for linux
        data_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False
        )
    else: # for windows
        batch_size = 1
        data_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    print('\nEvaluation loader information:\n' + str(data_loader.dataset))
    print('Evaluation batch size: {}\n'.format(batch_size))

    th2d_lst = np.array([i for i in range(1,50)])
    PCK2d_lst = np.zeros((len(th2d_lst),))
    mse2d_lst = np.zeros((21,))
    th3d_lst = np.array([i for i in range(1,51)])
    PCK3d_lst = np.zeros((len(th3d_lst),))
    mse3d_lst = np.zeros((21,))
    visibility_lst = np.zeros((21,))
    with torch.no_grad():
        start_time = time.time()
        pose2d_mse_loss = JointsMSELoss().cuda(args.gpu) if args.gpu != -1 else JointsMSELoss()
        pose3d_mse_loss = Joints3DMSELoss().cuda(args.gpu) if args.gpu != -1 else Joints3DMSELoss()

        infer_time = [0,0]
        start_time = time.time()
        n_valid = 0
        model.orig_img_size = inference_dataset.orig_img_size
        orig_width, orig_height = model.orig_img_size
        heatmap_size = cfg.MODEL.HEATMAP_SIZE

        for i, ret in enumerate(data_loader):
            # ori_imgs: b x 4 x 480 x 640 x 3
            # imgs: b x 4 x 3 x H x W
            # pose2d_gt: b x 4 x 21 x 2 (have not been transformed)
            # pose3d_gt: b x 21 x 3
            # visibility: b x 4 x 21
            # extrinsic matrix: b x 4 x 3 x 4 
            # intrinsic matrix: b x 3 x 3
            # if i < count: continue
            imgs = ret['imgs'].to(device)
            orig_imgs = ret['orig_imgs']
            pose2d_gt, pose3d_gt, visibility = ret['pose2d'], ret['pose3d'], ret['visibility']
            extrinsic_matrices, intrinsic_matrices = ret['extrinsic_matrices'], ret['intrinsic_matrix']
            # somtimes intrisic_matrix has a shape of 3x3 or b x 3x3
            intrinsic_matrix = intrinsic_matrices[0] if len(intrinsic_matrices.shape) == 3 else intrinsic_matrices
            
            batch_size = orig_imgs.shape[0]
            n_joints = pose2d_gt.shape[2]
            pose2d_gt = pose2d_gt.view(-1, *pose2d_gt.shape[2:]).numpy() # b*v x 21 x 2
            pose3d_gt = pose3d_gt.numpy() # b x 21 x 3
            visibility = visibility.view(-1, visibility.shape[2]).numpy()  # b*v x 21
            

            if 'pose_hrnet' in cfg.MODEL.NAME:
                s1 = time.time()
                heatmaps, _ = model(imgs.view(-1, *imgs.shape[2:])) # b*v x 21 x 64 x 64
                pose2d_pred = get_final_preds(heatmaps, cfg).view(batch_size, -1, n_joints, 2) # b x v x 21 x 2 NOTE: the estimated 2D poses are located in the heatmap size 64(W) x 64(H)
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices).to(device) # b x v x 3 x 4
                # rescale to the original image before DLT
                pose2d_pred[:,:,:,0:1] *= orig_width/heatmap_size[0]
                pose2d_pred[:,:,:,1:2] *= orig_height/heatmap_size[0]

                # 3D world coordinate 1 x 21 x 3
                pose3d_pred = torch.cat([DLT_sii_pytorch(pose2d_pred[:,:,k], proj_matrices).unsqueeze(1) for k in range(n_joints)], dim=1) # b x 21 x 3

                if i > 20:
                    infer_time[0] += 1
                    infer_time[1] += time.time() - s1
                    #print('FPS {:.1f}'.format(infer_time[0]/infer_time[1]))
            
            elif 'alg' == cfg.MODEL.NAME or 'ransac' == cfg.MODEL.NAME:
                s1 = time.time()
                # pose2d_pred: b x N_views x 21 x 2 
                # NOTE: the estimated 2D poses are located in the original image of size 640(W) x 480(H)]
                # pose3d_pred: b x 21 x 3 [world coord]
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices).to(device) # b x v x 3 x 4
                pose3d_pred,\
                pose2d_pred,\
                heatmaps,\
                confidences_pred = model(imgs.to(device), proj_matrices.to(device))
                if i > 20:
                    infer_time[0] += 1
                    infer_time[1] += time.time() - s1

            elif "vol" in cfg.MODEL.NAME:
                intrinsic_matrix = update_after_resize(
                    intrinsic_matrix,
                    (orig_height, orig_width),
                    tuple(heatmap_size))
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices).to(device) # b x v x 3 x 4
                s1 = time.time()

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
                    heatmaps_gt = ret['heatmaps']

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

                if i > 20:
                    infer_time[0] += 1
                    infer_time[1] += time.time() - s1

                pose2d_pred[:,:,:,0:1] *= orig_width/heatmap_size[0]
                pose2d_pred[:,:,:,1:2] *= orig_height/heatmap_size[1]

            # 2D errors
            pose2d_gt[:,:,0] *= orig_width / heatmap_size[0]
            pose2d_gt[:,:,1] *= orig_height / heatmap_size[1]
           
            pose2d_pred = pose2d_pred.view(-1, n_joints, 2).cpu().numpy() # b*v x 21 x 2
            for k in range(21):
                print(pose2d_gt[0,k].tolist(), pose2d_pred[0,k].tolist())
            input()
            mse_each_joint = np.linalg.norm(pose2d_pred - pose2d_gt, axis=2) * visibility # b*v x 21
            mse2d_lst += mse_each_joint.sum(axis=0)
            visibility_lst += visibility.sum(axis=0)

            for th_idx in range(len(th2d_lst)):
                PCK2d_lst[th_idx] += np.sum((mse_each_joint < th2d_lst[th_idx]) * visibility)
            
            # 3D errors
            for k in range(21):
                print(pose3d_gt[0,k].tolist(), pose3d_pred[0,k].tolist())
            input()
            visibility = visibility.reshape((batch_size, -1, n_joints)) # b x v x 21
            for b in range(batch_size):
                # print(np.sum(visibility[b]), visibility[b].size)
                if np.sum(visibility[b]) >= visibility[b].size * 0.65:
                    n_valid += 1
                    mse_each_joint = np.linalg.norm(pose3d_pred[b].cpu().numpy() - pose3d_gt[b], axis=1) # 21
                    mse3d_lst += mse_each_joint

                    for th_idx in range(len(th3d_lst)):
                        PCK3d_lst[th_idx] += np.sum(mse_each_joint < th3d_lst[th_idx])

            if i % (len(data_loader)//5) == 0:
                    print("[Evaluation]{}% finished.".format(20 * i // (len(data_loader)//5)))
            #if i == 10:break
        print('Evaluation spent {:.2f} s\tFPS: {:.1f}'.format(time.time()-start_time, infer_time[0]/infer_time[1]))

        mse2d_lst /= visibility_lst
        PCK2d_lst /= visibility_lst.sum()
        mse3d_lst /= n_valid
        PCK3d_lst /= (n_valid * 21)
        plot_performance(PCK2d_lst, th2d_lst, PCK3d_lst, th3d_lst, mse2d_lst, mse3d_lst)

        if not os.path.exists(result):
            os.mkdir(result)
        result_dir = prefix+cfg.EXP_NAME
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        np.savetxt(os.path.join(result_dir, 'mse2d_each_joint.txt'), mse2d_lst, fmt='%.4f')
        np.savetxt(os.path.join(result_dir, 'mse3d_each_joint.txt'), mse3d_lst, fmt='%.4f')
        np.savetxt(os.path.join(result_dir, 'PCK2d.txt'), np.stack((th2d_lst, PCK2d_lst)))
        np.savetxt(os.path.join(result_dir, 'PCK3d.txt'), np.stack((th3d_lst, PCK3d_lst)))

main()