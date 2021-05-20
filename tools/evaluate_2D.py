from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn

import _init_paths
from config import cfg
from config import update_config

from utils.utils import get_model_summary
from ptflops import get_model_complexity_info
from fp16_utils.fp16util import network_to_half
from core.loss import BoneLengthLoss, JointAngleLoss, JointsMSELoss
import dataset

from dataset.frei_utils.fh_utils import *
from dataset.HandGraph_utils.vis import *
from dataset import build_transforms
from models import pose_hrnet, pose_hrnet_softmax, pose_hrnet_hamburger,swin_transformer, pose_hrnet_transformer, pose_hrnet_PoseAggr,pose_hrnet_volumetric, multiview_pose_hrnet, CPM
from utils.heatmap_decoding import get_final_preds
from utils.misc import plot_performance
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
    parser.add_argument('--batch_size',
                    default=32,
                    type=int)
    parser.add_argument('--model_path', default='', type=str)

    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    
    update_config(cfg, args)
    cfg.defrost()
    cfg.freeze()
    
    record_prefix = './eval2D_results_'
    if args.is_vis:
        result_dir = record_prefix + cfg.EXP_NAME
        mse2d_lst = np.loadtxt(os.path.join(result_dir, 'mse2d_each_joint.txt'))
        PCK2d_lst = np.loadtxt(os.path.join(result_dir, 'PCK2d.txt'))

        plot_performance(PCK2d_lst[1,:], PCK2d_lst[0,:], mse2d_lst)
        exit()

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_path = args.model_path
    is_vis = args.is_vis
    
    # FP16 SETTING
    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")
    
    model = eval(cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)

    # # calculate GFLOPS
    # dump_input = torch.rand(
    #     (5, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[0])
    # )
    
    # print(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    # ops, params = get_model_complexity_info(
    #    model, (3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[0]),
    #    as_strings=True, print_per_layer_stat=True, verbose=True)
    # input()

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not args.distributed:
        print('Warning: Sync BatchNorm is only supported in distributed training.')

    if args.gpu != -1:
        device = torch.device('cuda:'+str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    # load model state
    if model_path:
        print("Loading model:", model_path)
        ckpt = torch.load(model_path)#, map_location='cpu')
        if 'state_dict' not in ckpt.keys():
            state_dict = ckpt
        else:
            state_dict = ckpt['state_dict']
            print('Model epoch {}'.format(ckpt['epoch']))
        
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        
        model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    
    # calculate GFLOPS
    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[0])
    ).to(device)
    
    print(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    model.eval()

    # inference_dataset = eval('dataset.{}'.format(cfg.DATASET.TEST_DATASET[0].replace('_kpt','')))(
    #     cfg.DATA_DIR,
    #     cfg.DATASET.TEST_SET,
    #     transform=transform
    # )
    inference_dataset = eval('dataset.{}'.format(cfg.DATASET.TEST_DATASET[0].replace('_kpt','')))(
        cfg.DATA_DIR,
        cfg.DATASET.TEST_SET,
        transforms=build_transforms(cfg, is_train=False)
    )

    batch_size = args.batch_size
    data_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=batch_size, #48
        shuffle=False,
        num_workers=min(8, batch_size), #8
        pin_memory=False
    )

    print('\nEvaluation loader information:\n' + str(data_loader.dataset))
    n_joints = cfg.DATASET.NUM_JOINTS
    th2d_lst = np.array([i for i in range(1,50)])
    PCK2d_lst = np.zeros((len(th2d_lst),))
    mse2d_lst = np.zeros((n_joints,))
    visibility_lst = np.zeros((n_joints,))

    print('Start evaluating... [Batch size: {}]\n'.format(data_loader.batch_size))
    with torch.no_grad():
        pose2d_mse_loss = JointsMSELoss().to(device)
        infer_time = [0,0]
        start_time = time.time()
        for i, ret in enumerate(data_loader):
            # pose2d_gt: b x 21 x 2 is [u,v] 0<=u<64, 0<=v<64 (heatmap size)
            # visibility: b x 21 vis=0/1
            imgs = ret['imgs']
            pose2d_gt = ret['pose2d'] # b [x v] x 21 x 2
            visibility = ret['visibility'] # b [x v] x 21 x 1

            s1 = time.time()
            if 'CPM' == cfg.MODEL.NAME:
                pose2d_gt = pose2d_gt.view(-1, *pose2d_gt.shape[-2:])
                heatmap_lst = model(imgs.to(device), ret['centermaps'].to(device)) # 6 groups of heatmaps, each of which has size (1,22,32,32)
                heatmaps = heatmap_lst[-1][:,1:]
                pose2d_pred = data_loader.dataset.get_kpts(heatmaps)
                hm_size = heatmap_lst[-1].shape[-1] # 32
            else:
                if cfg.MODEL.NAME == 'pose_hrnet_transformer':
                    # imgs: b(1) x (4*seq_len) x 3 x 256 x 256
                    n_batches, seq_len = imgs.shape[0], imgs.shape[1] // 4
                    idx_lst = torch.tensor([4 * i for i in range(seq_len)])
                    imgs = torch.stack(
                        [imgs[b, idx_lst + cam_idx] for b in range(n_batches) for cam_idx in range(4)]
                    ) # (b*4) x seq_len x 3 x 256 x 256

                    pose2d_pred, heatmaps_pred, _ = model(imgs.cuda(device)) # (b*4) x 21 x 2
                    pose2d_gt = pose2d_gt[:,4*(seq_len//2):4*(seq_len//2+1)].contiguous().view(-1, *pose2d_pred.shape[-2:]) # (b*4) x 21 x 2
                    visibility = visibility[:,4*(seq_len//2):4*(seq_len//2+1)].contiguous().view(-1, *visibility.shape[-2:]) # (b*4) x 21
    
                else:
                    if 'Aggr' in cfg.MODEL.NAME:
                        # imgs: b x (4*5) x 3 x 256 x 256
                        n_batches, seq_len = imgs.shape[0], len(cfg.DATASET.SEQ_IDX)
                        true_batch_size = imgs.shape[1] // seq_len
                        pose2d_gt = torch.cat(
                            [pose2d_gt[b,true_batch_size*(seq_len//2):true_batch_size*(seq_len//2+1)] for b in range(n_batches)],
                            dim=0)
        
                        visibility = torch.cat(
                            [visibility[b,true_batch_size*(seq_len//2):true_batch_size*(seq_len//2+1)] for b in range(n_batches)],
                            dim=0)

                        imgs = torch.cat(
                            [imgs[b,true_batch_size*j:true_batch_size*(j+1)] for j in range(seq_len) for b in range(n_batches)],
                            dim=0) # (b*4*5) x 3 x 256 x 256

                        heatmaps_pred, _ = model(imgs.to(device))
                    else:
                        pose2d_gt = pose2d_gt.view(-1, *pose2d_gt.shape[-2:])
                        heatmaps_pred, _ = model(imgs.to(device)) # b x 21 x 64 x 64
                
                    pose2d_pred = get_final_preds(heatmaps_pred, cfg.MODEL.HEATMAP_SOFTMAX) # b x 21 x 2
                
                hm_size = heatmaps_pred.shape[-1] # 64

            if i > 20:
                infer_time[0] += 1
                infer_time[1] += time.time() - s1

            # rescale to the original image before DLT
            
            if 'RHD' in cfg.DATASET.TEST_DATASET[0]:
                crop_size, corner = ret['crop_size'], ret['corner']
                crop_size, corner = crop_size.view(-1, 1, 1), corner.unsqueeze(1) # b x 1 x 1; b x 2 x 1
                pose2d_pred = pose2d_pred.cpu() * crop_size/hm_size + corner
                pose2d_gt = pose2d_gt * crop_size/hm_size + corner
            else:
                orig_width, orig_height = data_loader.dataset.orig_img_size
                pose2d_pred[:,:,0] *= orig_width/hm_size
                pose2d_pred[:,:,1] *= orig_height/hm_size
                pose2d_gt[:,:,0] *= orig_width/hm_size
                pose2d_gt[:,:,1] *= orig_height/hm_size
            
                # for k in range(21):
                #     print(pose2d_gt[0,k].tolist(), pose2d_pred[0,k].tolist())
                # input()
            # 2D errors
            pose2d_pred, pose2d_gt, visibility = pose2d_pred.cpu().numpy(), pose2d_gt.numpy(), visibility.squeeze(2).numpy()

            # import matplotlib.pyplot as plt
            # imgs = cv2.resize(imgs[0].permute(1,2,0).cpu().numpy(), tuple(data_loader.dataset.orig_img_size))
            # for k in range(21):
            #     print(pose2d_gt[0,k],pose2d_pred[0,k],visibility[0,k])
            # for k in range(0,21,5):
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(131)
            #     ax2 = fig.add_subplot(132)
            #     ax3 = fig.add_subplot(133)
            #     ax1.imshow(cv2.cvtColor(imgs / imgs.max(), cv2.COLOR_BGR2RGB))
            #     plot_hand(ax1, pose2d_gt[0,:,0:2], order='uv')
            #     ax2.imshow(cv2.cvtColor(imgs / imgs.max(), cv2.COLOR_BGR2RGB))
            #     plot_hand(ax2, pose2d_pred[0,:,0:2], order='uv')
            #     ax3.imshow(heatmaps_pred[0,k].cpu().numpy())
            #     plt.show()
            mse_each_joint = np.linalg.norm(pose2d_pred - pose2d_gt, axis=2) * visibility # b x 21

            mse2d_lst += mse_each_joint.sum(axis=0)
            visibility_lst += visibility.sum(axis=0)

            for th_idx in range(len(th2d_lst)):
                PCK2d_lst[th_idx] += np.sum((mse_each_joint < th2d_lst[th_idx]) * visibility)
            
            period = 10
            if i % (len(data_loader)//period) == 0:
                    print("[Evaluation]{}% finished.".format(period * i // (len(data_loader)//period)))
            #if i == 10:break
        print('Evaluation spent {:.2f} s\tfps: {:.1f} {:.4f}'.format(time.time()-start_time, infer_time[0]/infer_time[1], infer_time[1]/infer_time[0]))

        mse2d_lst /= visibility_lst
        PCK2d_lst /= visibility_lst.sum()

        result_dir = record_prefix+cfg.EXP_NAME
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        
        mse_file, pck_file = os.path.join(result_dir, 'mse2d_each_joint.txt'), os.path.join(result_dir, 'PCK2d.txt')
        print('Saving results to ' + mse_file)
        print('Saving results to ' + pck_file)
        np.savetxt(mse_file, mse2d_lst, fmt='%.4f')
        np.savetxt(pck_file, np.stack((th2d_lst, PCK2d_lst)))

        plot_performance(PCK2d_lst, th2d_lst, mse2d_lst)

main()