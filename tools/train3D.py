# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# python train.py --cfg ../experiments/HandGraph/HG_w32_256x256_adam_lr1e-3.yaml
# python train.py --cfg ../experiments/RHD/RHD_w32_256x256_adam_lr1e-3.yaml
# python train.py --cfg ../experiments/FreiHand/Frei_w32_256x256_adam_lr1e-3.yaml
# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 train.py --cfg ../experiments/FreiHand/Frei_w32_256x256_adam_lr1e-3.yaml
# --cfg ../experiments/MHP/MHP_v1.yaml
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import time
import warnings
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed.optim import DistributedOptimizer
# from torch.distributed.rpc import RRef
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import cv2

import _init_paths
from config import cfg
from config import update_config
from core.loss import *
from core.function3D import train
from core.function3D import validate
from utils.utils import get_optimizer
from utils.utils import create_logger
from utils.utils import setup_logger
from utils.utils import get_model_summary
from fp16_utils.fp16util import network_to_half
from fp16_utils.fp16_optimizer import FP16_Optimizer

import dataset
from dataset import make_dataloader, make_test_dataloader
import models
from models.FTL_encoder_decoder import FTLMultiviewNet
from models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet, VolumetricTriangulationNet_CPM


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpus',
                        help='gpus id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    # For DDP
    parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()
    
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    #logger.info(cfg)

    if cfg.WITHOUT_EVAL:
        input("[WARNING] According to the configuration, there will be no evaluation. If evaluation is necessary, please terminate this process. [press Enter to continue]")
        logger.info("=> Training without evaluation")

    ngpus_per_node = len(cfg.GPUS)
    if ngpus_per_node == 1:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
    
    # Simply call main_worker function
    main_worker(
        ','.join([str(i) for i in cfg.GPUS]),
        ngpus_per_node,
        args,
        final_output_dir,
        tb_log_dir
    )

def main_worker(gpus, ngpus_per_node, args, final_output_dir, tb_log_dir):
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    #os.environ['CUDA_VISIBLE_DEVICES']=gpus
    
    # if len(gpus) == 1:
    #     gpus = int(gpus) 


    update_config(cfg, args)
    
    #test(cfg, args)

    # logger setting
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # model initilization
    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet,
        "vol_CPM": VolumetricTriangulationNet_CPM,
        "FTL": FTLMultiviewNet
    }[cfg.MODEL.NAME](cfg)

    # load pretrained model before DDP initialization
    if cfg.AUTO_RESUME:
        checkpoint_file = os.path.join(final_output_dir, 'model_best.pth.tar')
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']

            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)
            
            model.load_state_dict(state_dict)
            logger.info("=> Loading checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
        else:
            print('[Warning] Checkpoint file not found! Wrong path: {}'.format(checkpoint_file))

    elif cfg.MODEL.HRNET_PRETRAINED:
        logger.info("=> loading a pretrained model '{}'".format(cfg.MODEL.PRETRAINED))
        checkpoint = torch.load(cfg.MODEL.HRNET_PRETRAINED)

        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        
        model.load_state_dict(state_dict)
    
    # initiliaze a optimizer
    # optimizer must be initilized after model initilization
    if cfg.MODEL.TRIANGULATION_MODEL_NAME == "vol":
        optimizer = torch.optim.Adam(
            [{'params': model.backbone.parameters(), 'initial_lr': cfg.TRAIN.LR},
            {'params': model.process_features.parameters(), 'initial_lr': cfg.TRAIN.PROCESS_FEATURE_LR if hasattr(cfg.TRAIN, "PROCESS_FEATURE_LR") else cfg.TRAIN.LR},
            {'params': model.volume_net.parameters(), 'initial_lr': cfg.TRAIN.VOLUME_NET_LR if hasattr(cfg.TRAIN, "VOLUME_NET_LR") else cfg.TRAIN.LR}
            ],
            lr=cfg.TRAIN.LR
        )
    else:
        optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': cfg.TRAIN.LR}], lr=cfg.TRAIN.LR)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models',  'triangulation.py'),
        final_output_dir
    )
    # copy configuration file
    config_dir = args.cfg
    shutil.copy2(os.path.join(args.cfg),
        final_output_dir)    

    # calculate GFLOPS
    # dump_input = torch.rand(
    #     (1, 4, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[0])
    # )
    
    # logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    # FP16 SETTING
    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")
    
    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not cfg.DISTRIBUTED:
        print('Warning: Sync BatchNorm is only supported in distributed training.')

    if cfg.FP16.ENABLED:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=cfg.FP16.STATIC_LOSS_SCALE,
            dynamic_loss_scale=cfg.FP16.DYNAMIC_LOSS_SCALE,
            verbose=False
        )
    
    # Distributed Computing
    master = True
    if cfg.DISTRIBUTED: # This block is not available
        args.local_rank+=int(gpus[0])
        print('This process is using GPU', args.local_rank)
        device = args.local_rank
        master = device == int(gpus[0])
        dist.init_process_group(backend='nccl')
        if cfg.MODEL.SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpus is not None:
            torch.cuda.set_device(device)
            model.cuda(device)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # workers = int(workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device],
                output_device=device,
                find_unused_parameters=True
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    else: # implement this block
        gpu_ids = eval('['+gpus+']')
        device = gpu_ids[0]
        print('This process is using GPU', str(device))
        model = torch.nn.DataParallel(model, gpu_ids).cuda(device)

    # Prepare loss functions
    criterion = {}
    if cfg.LOSS.WITH_HEATMAP_LOSS:
        criterion['heatmap_loss'] = HeatmapLoss().cuda(device)
    if cfg.LOSS.WITH_POSE2D_LOSS:
        criterion['pose2d_loss'] = JointsMSELoss().cuda(device)
    if cfg.LOSS.WITH_POSE3D_LOSS:
        criterion['pose3d_loss'] = Joints3DMSELoss().cuda(device)
    if cfg.LOSS.WITH_VOLUMETRIC_CE_LOSS:
        criterion['volumetric_ce_loss'] = VolumetricCELoss().cuda(device)
    if cfg.LOSS.WITH_BONE_LOSS:
        criterion['bone_loss'] = BoneLengthLoss().cuda(device)
    if cfg.LOSS.WITH_TIME_CONSISTENCY_LOSS:
        criterion['time_consistency_loss'] = Joints3DMSELoss().cuda(device)
    if cfg.LOSS.WITH_KCS_LOSS:
        criterion['KCS_loss'] = HeatmapLoss(mode='l1').cuda(device)
    if cfg.LOSS.WITH_JOINTANGLE_LOSS:
        criterion['jointangle_loss'] = JointAngleLoss().cuda(device)
    
    best_perf = 1e9
    best_model = False
    last_epoch = -1
    
    # load history
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        begin_epoch = checkpoint['epoch'] + 1
        best_perf = checkpoint['loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'train_global_steps' in checkpoint.keys() and \
        'valid_global_steps' in checkpoint.keys():
            writer_dict['train_global_steps'] = checkpoint['train_global_steps']
            writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']

    # Floating point 16 mode
    if cfg.FP16.ENABLED:
        logger.info("=> Using FP16 mode")
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer.optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=begin_epoch
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=begin_epoch
        )

    # Data loading code
    train_loader_dict = make_dataloader(cfg, is_train=True, distributed=cfg.DISTRIBUTED)
    valid_loader_dict = make_dataloader(cfg, is_train=False, distributed=cfg.DISTRIBUTED)

    for i, (dataset_name, train_loader) in enumerate(train_loader_dict.items()):
        logger.info('Training Loader {}/{}:\n'.format(i+1, len(train_loader_dict)) + str(train_loader.dataset))
    for i, (dataset_name, valid_loader) in enumerate(valid_loader_dict.items()):
        logger.info('Validation Loader {}/{}:\n'.format(i+1, len(valid_loader_dict)) + str(valid_loader.dataset))

    #writer_dict['writer'].add_graph(model, (dump_input, ))
    """
    Start training
    """
    start_time = time.time()

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
            epoch_start_time = time.time()
            # shuffle datasets with the sample random seed
            if cfg.DISTRIBUTED:
                for data_loader in train_loader_dict.values():
                    data_loader.sampler.set_epoch(epoch)
            # train for one epoch
            logger.info('Start training [{}/{}]'.format(epoch, cfg.TRAIN.END_EPOCH-1))
            train(epoch, cfg, args, master, train_loader_dict, model, criterion, optimizer,
                final_output_dir, tb_log_dir, writer_dict, logger, device, fp16=cfg.FP16.ENABLED)

            # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
            lr_scheduler.step()

            # evaluate on validation set
            if not cfg.WITHOUT_EVAL:
                logger.info('Start evaluating [{}/{}]'.format(epoch, cfg.TRAIN.END_EPOCH-1))
                with torch.no_grad():
                    recorder = validate(
                        cfg, args, master, valid_loader_dict, model, criterion,
                        final_output_dir, tb_log_dir, writer_dict, logger, device
                    )

                val_total_loss = recorder.avg_total_loss

                if val_total_loss < best_perf:
                    logger.info('This epoch yielded a better model with total loss {:.4f} < {:.4f}.'.format(val_total_loss, best_perf))
                    best_perf = val_total_loss
                    best_model = True
                else:
                    best_model = False

            else:
                val_total_loss = 0
                best_model = True

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch,
                'model': cfg.EXP_NAME + '.' + cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'loss': val_total_loss,
                'optimizer': optimizer.state_dict(),
                'train_global_steps': writer_dict['train_global_steps'],
                'valid_global_steps': writer_dict['valid_global_steps']
            }, best_model, final_output_dir)

            print('\nEpoch {} spent {:.2f} hours\n'.format(epoch, (time.time()-epoch_start_time) / 3600))

            #if epoch == 3:break
    if master:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state{}.pth.tar'.format(gpus)
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        torch.save(model.state_dict(), final_model_state_file)
        writer_dict['writer'].close()

        print('\n[Training Accomplished] {} epochs spent {:.2f} hours\n'.format(cfg.TRAIN.END_EPOCH - begin_epoch + 1, (time.time()-start_time) / 3600))

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states,os.path.join(output_dir, 'model_best.pth.tar'))
        torch.save(states['state_dict'], os.path.join(output_dir, 'best_state_epoch{}.pth.tar'.format(states['epoch'])))

def test(cfg, args):
        # train_dataset = dataset.HandGraph(cfg.DATASET.ROOT,
        # cfg.DATASET.TRAIN_SET,
        # 'png')
        # train_dataset.visualize_data()
        train_dataset = make_dataloader(cfg, is_train=True).dataset
        train_dataset.visualize_data()

def visualize_samples(config, data_loader):
    import matplotlib.pyplot as plt
    import numpy as np
# visualize some samples
    dataset = data_loader.dataset
    print(type(dataset),dataset)
    dataset.visualize_data()
    
if __name__ == '__main__':
    main()
