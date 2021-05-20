from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.misc import update_after_resize, project3Dto2D
from utils.transforms import flip_back, scale_pose3d, scale_pose2d
from dataset.standard_legends import KC_matrix

debug = False

def train(epoch, config, args, master, train_loader_dict, model, criterion, optimizer,
          output_dir, tb_log_dir, writer_dict, logger, device, fp16=False):
    """
    - Brief: Training phase
    - params:
        config
        train_loader:
        model:
        criterion: a dict containing loss items
        device: the index of a GPU
    """
    recorder = AverageMeter(config, criterion)
    writer = writer_dict['writer']
    # switch to train mode
    model.train()

    for dataset_name, train_loader in train_loader_dict.items():
        logger.info('Training on {} dataset [Batch size: {}]\n'.format(dataset_name, train_loader.batch_size))
        
        end = time.time()
        orig_width, orig_height = train_loader.dataset.orig_img_size
        new_img_size = config.MODEL.IMAGE_SIZE # [256,256]
        heatmap_size = config.MODEL.HEATMAP_SIZE # [64,64]

        for i, ret in enumerate(train_loader):
            # ori_imgs: b x N_views x H(480) x W(640) x 3
            # imgs: b x N_views x 3 x H x W
            # pose2d_gt: b x N_views x 21 x 2 [located in heatmaps of size 64 x 64]
            # pose3d_gt: b x 21 x 3
            # keypoints_2d_visibility: b x N_views x 21 x 1
            # extrinsic_matrices (H): b x N_views x 3 x 4
            # intrinsic_matrices (K): b x 3 x 3 [Note: it's not been scaled]
            if hasattr(train_loader.dataset, 'exception'):
                if train_loader.dataset.exception:
                    continue
            imgs, pose2d_gt, visibility, pose3d_gt = ret['imgs'], ret['pose2d'], ret['visibility'], ret['pose3d'].cuda(device, non_blocking=True)
            extrinsic_matrices, intrinsic_matrices = ret['extrinsic_matrices'], ret['intrinsic_matrix']
            # somtimes intrisic_matrix has a shape of 3x3 or b x 3x3
            intrinsic_matrix = intrinsic_matrices[0] if len(intrinsic_matrices.shape) == 3 else intrinsic_matrices

            model_type = config.MODEL.NAME
            
            if model_type == "FTL":
                b, v = extrinsic_matrices.shape[0:2]
                # pose2d_pred:
                # pose3d_pred:
                heatmaps, pose2d_pred, pose3d_pred = model(imgs, extrinsic_matrices, intrinsic_matrices)
            elif model_type == "alg" or model_type == "ransac":
                # heatmaps: b x v x 21 x 64 x 64
                # pose2d_pred: b x v x 21 x 2 [located in the original image of size 640(W) x 480(H)]
                # pose3d_pred: b x 21 x 3 [world coord]
                pose2d_gt[:,:,:,0] = pose2d_gt[:,:,:,0] * orig_width / heatmap_size[0]
                pose2d_gt[:,:,:,1] = pose2d_gt[:,:,:,1] * orig_height / heatmap_size[1]
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices) # b x v x 3 x 4

                pose3d_pred,\
                pose2d_pred,\
                heatmaps_pred,\
                confidences_pred = model(imgs.cuda(device, non_blocking=True), proj_matrices.cuda(device, non_blocking=True), orig_img_size=train_loader.dataset.orig_img_size)

            elif "vol" in model_type:
                # pose3d_pred (torch.tensor) b x 21 x 3
                # pose2d_pred (torch.tensor) b x v x 21 x 2
                # heatmaps_pred (torch.tensor) b x v x 21 x 64 x 64
                # volumes_pred (torch.tensor)
                # confidences_pred (torch.tensor)
                # cuboids_pred (list)
                # coord_volumes_pred (torch.tensor)
                # base_points_pred (torch.tensor) b x v x 1 x 2

                # the intrinsic matrix needs to be scaled so that it represents 64 x 64 images
                intrinsic_matrix = update_after_resize(
                    intrinsic_matrix,
                    (orig_height, orig_width),
                    tuple(heatmap_size))
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices) # b x v x 3 x 4
                if config.MODEL.BACKBONE_NAME == 'CPM_volumetric':
                    centermaps = ret['centermaps']
                    heatmaps_gt = ret['heatmaps']

                    pose3d_pred,\
                    pose2d_pred,\
                    heatmaps_pred,\
                    volumes_pred,\
                    confidences_pred,\
                    coord_volumes_pred,\
                    base_points_pred\
                        = model(imgs.cuda(device, non_blocking=True), centermaps.cuda(device, non_blocking=True), proj_matrices.cuda(device, non_blocking=True))

                else:
                    pose3d_pred,\
                    pose2d_pred,\
                    heatmaps_pred,\
                    volumes_pred,\
                    confidences_pred,\
                    coord_volumes_pred,\
                    base_points_pred\
                        = model(imgs.cuda(device, non_blocking=True), proj_matrices.cuda(device, non_blocking=True))

            batch_size, n_views= imgs.shape[0], imgs.shape[1]
            n_joints = pose3d_pred.shape[1]
            #pose3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)
            scale_keypoints_3d = config.MODEL.SCALE_KEYPOINTS_3D if hasattr(config.MODEL, "SCALE_KEYPOINTS_3D") else 1.0

            # 1-view case
            if n_views == 1:
                base_joint = 9 # the middle finger root is specified as the cuboid center

                keypoints_3d_gt_transformed = pose3d_gt.clone()
                keypoints_3d_gt_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_gt_transformed[:, base_joint:base_joint + 1]
                pose3d_gt = keypoints_3d_gt_transformed

                keypoints_3d_pred_transformed = pose3d_pred.clone()
                keypoints_3d_pred_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_pred_transformed[:, base_joint:base_joint + 1]
                pose3d_pred = keypoints_3d_pred_transformed

            # calculate losses
            if model_type == "FTL":
                if 1 <= epoch <= 20:
                    loss_dict = recorder.computeLosses(
                        pose2d_pred=pose2d_pred.view(-1,21,2),
                        pose2d_gt=pose2d_gt.view(-1,21,2).cuda(device))
                elif 20 < epoch:
                    # pose3d_gt_rel = torch.zeros(pose3d_gt.shape).float()
                    # pose3d_gt_rel[:,1:,:] = pose3d_gt[:,1:,:] - pose3d_gt[:,0:1,:]
                    # pose3d_gt_pred = torch.zeros(pose3d_pred.shape).float()
                    # pose3d_gt_pred[:,1:,:] = pose3d_pred[:,1:,:] - pose3d_pred[:,0:1,:]
                    loss_dict = recorder.computeLosses(
                        pose2d_pred=0.1*pose2d_pred.view(-1,21,2),
                        pose2d_gt=0.1*pose2d_gt.view(-1,21,2).cuda(device),
                        pose3d_pred=pose3d_pred,
                        pose3d_gt=pose3d_gt.cuda(device))
            else:
                if False:
                    # pose3d b x v x 21 x 3
                    pose2d_pred_reproj = torch.cat([project3Dto2D(pose3d_pred[:,k], proj_matrices).unsqueeze(2) for k in range(pose3d_pred.shape[1])], dim=2)# 
                # print(pose2d_gt.shape, pose2d_pred.shape)
                # for k in range(21):
                #     print(pose2d_gt.view(-1,21,2)[0,k].tolist(), pose2d_pred.view(-1,21,2)[0,k].tolist())
                # print(torch.norm(pose2d_gt.view(-1,21,2) - pose2d_pred.view(-1,21,2).cpu(), dim=2).sum() / pose2d_pred.view(-1,21,2).shape[1])
                # input()
                item_dict = {
                    'pose3d_pred': pose3d_pred, # b x 21 x 3
                    'pose3d_gt': pose3d_gt,
                    'pose3d_binary_validity_gt': torch.ones((batch_size, n_joints, 1), dtype=torch.float32, device=device),  # 源代码中该参数被设置为全1.0的张量，见human36m.py的172行
                }
                if 'vol' in model_type:
                    item_dict['coord_volumes_pred'] = coord_volumes_pred
                    item_dict['volumes_pred'] = volumes_pred
                if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                    item_dict['data_idx'] = ret['data_idx'] # (b,)
                if config.LOSS.WITH_HEATMAP_LOSS:
                    item_dict['heatmaps_gt'] = heatmaps_gt.cuda(device, non_blocking=True)
                    item_dict['heatmaps_pred'] = heatmaps_pred
                if config.LOSS.WITH_KCS_LOSS:
                    global KC_matrix
                    KC_matrix = KC_matrix.cuda(device, non_blocking=True) # 20 x 21
                    kinematic_chain_gt = KC_matrix @ pose3d_gt # b x 20 x 3

                    kinematic_chain_pred = KC_matrix @ pose3d_pred # b x 20 x 3
                    kinematic_chain_Pred_T = kinematic_chain_pred.clone().transpose(1,2) # b x 3 x 20

                    item_dict['KCS_gt'] =  kinematic_chain_gt @ kinematic_chain_gt.transpose(1,2) # b x 20 x 20
                    item_dict['KCS_pred'] = kinematic_chain_pred @ kinematic_chain_Pred_T
                    if config.LOSS.WITH_KCS_TC_LOSS:
                        item_dict['data_idx'] = ret['data_idx'] # (b,)
                if config.LOSS.WITH_POSE2D_LOSS:
                    item_dict['pose2d_pred'] = pose2d_pred.view(-1,21,2)
                    item_dict['pose2d_gt'] = pose2d_gt.view(-1,21,2).cuda(device, non_blocking=True)
                    item_dict['pose2d_visibility'] = visibility.view(-1,21).cuda(device, non_blocking=True)

                loss_dict = recorder.computeLosses(item_dict)

            total_loss = loss_dict['total_loss']
            heatmap_loss = loss_dict['heatmap_loss']
            pose2d_loss = loss_dict['pose2d_loss']
            pose3d_loss = loss_dict['pose3d_loss'] # normal value: 100-200
            volumetric_ce_loss = loss_dict['volumetric_ce_loss']
            KCS_loss = loss_dict['KCS_loss']
            KCS_TC_loss = loss_dict['KCS_TC_loss']
            time_consistency_loss = loss_dict['time_consistency_loss']

            # compute gradient and do update step
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(total_loss)
            else:
                total_loss.backward()
            optimizer.step()

            if master and i % config.PRINT_FREQ == 0:
                recorder.computeAvgLosses()
                # measure elapsed time
                batch_time = (time.time() - end) / config.PRINT_FREQ
                end = time.time()

                global_steps = writer_dict['train_global_steps']

                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time:.3f}s\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})\t'.format(
                        epoch, i+1, len(train_loader),
                        batch_time=batch_time,
                        speed=imgs.shape[0]/batch_time,
                        total_loss=total_loss.item(),
                        total_loss_avg=recorder.avg_total_loss)

                if heatmap_loss:
                    msg += '\tHeatmapLoss {heatmap_loss:.5f} ({heatmap_loss_avg:.5f})'.format(
                        heatmap_loss = heatmap_loss.item(), heatmap_loss_avg = recorder.avg_heatmap_loss
                    )
                    writer.add_scalar('train_loss/heatmap_loss', heatmap_loss, global_steps)
                if pose2d_loss:
                    msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                        pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
                    )
                    writer.add_scalar('train_loss/pose2d_loss', pose2d_loss, global_steps)
                if pose3d_loss:
                    msg += '\tPose3DLoss {pose3d_loss:.5f} ({pose3d_loss_avg:.5f})'.format(
                        pose3d_loss = pose3d_loss.item(), pose3d_loss_avg = recorder.avg_pose3d_loss
                    )
                    writer.add_scalar('train_loss/pose3d_loss', pose3d_loss, global_steps)
                if time_consistency_loss:
                    msg += '\tTime_consistency_loss {time_consistency_loss:.5f} ({time_consistency_loss_avg:.5f})'.format(
                        time_consistency_loss = time_consistency_loss.item(), time_consistency_loss_avg = recorder.avg_time_consistency_loss
                    )
                    writer.add_scalar('train_loss/time_consistency_loss', time_consistency_loss, global_steps)
                if KCS_loss:
                    msg += '\tKCS_loss {KCS_loss:.5f} ({KCS_loss_avg:.5f})'.format(
                        KCS_loss = KCS_loss.item(), KCS_loss_avg = recorder.avg_KCS_loss
                    )
                    writer.add_scalar('train_loss/KCS_loss', KCS_loss, global_steps)
                if KCS_TC_loss:
                    msg += '\tKCS_TC_loss {KCS_TC_loss:.5f} ({KCS_TC_loss_avg:.5f})'.format(
                        KCS_TC_loss = KCS_TC_loss.item(), KCS_TC_loss_avg = recorder.avg_KCS_TC_loss
                    )
                    writer.add_scalar('train_loss/KCS_TC_loss', KCS_TC_loss, global_steps)
                if volumetric_ce_loss:
                    msg += '\tVolumetricCELoss {volumetric_ce_loss:.5f} ({volumetric_ce_loss_avg:.5f})'.format(
                        volumetric_ce_loss = volumetric_ce_loss.item(), volumetric_ce_loss_avg = recorder.avg_volumetric_ce_loss
                    )
                    writer.add_scalar('train_loss/volumetric_ce_loss', volumetric_ce_loss, global_steps)
              
                logger.info(msg)
                    
                writer.add_scalar('train_loss/total_loss', total_loss, global_steps)
                writer_dict['train_global_steps'] += 1

            if debug and i==3:break 

    return recorder

def validate(config, args, master, val_loader_dict, model, criterion, output_dir,
             tb_log_dir, writer_dict, logger, device):
    recorder = AverageMeter(config, criterion)
    writer = writer_dict['writer']
    # switch to evaluate mode
    model.eval()
    
    for dataset_name, val_loader in val_loader_dict.items():
        logger.info('Validating on {} dataset [Batch size: {}]\n'.format(dataset_name, val_loader.batch_size))
        end = time.time()
        orig_width, orig_height = val_loader.dataset.orig_img_size
        new_img_size = config.MODEL.IMAGE_SIZE # [256,256]
        heatmap_size = config.MODEL.HEATMAP_SIZE # [64,64]

        for i,  ret in enumerate(val_loader):
            # ori_imgs: b x N_views x H(480) x W(640) x 3
            # imgs: b x N_views x 3 x H x W
            # pose2d_gt: b x N_views x 21 x 2 [located in heatmaps of size 64 x 64]
            # pose3d_gt: b x 21 x 3
            # keypoints_2d_visibility: b x N_views x 21 x 1
            # extrinsic_matrices (H): b x N_views x 3 x 4
            # intrinsic_matrices (K): b x 3 x 3 [Note: it's not been scaled]
            if hasattr(val_loader.dataset, 'exception'):
                if val_loader.dataset.exception:
                    continue

            imgs, pose2d_gt, visibility, pose3d_gt = ret['imgs'], ret['pose2d'], ret['visibility'], ret['pose3d'].cuda(device, non_blocking=True)
            extrinsic_matrices, intrinsic_matrices = ret['extrinsic_matrices'], ret['intrinsic_matrix']
            # somtimes intrisic_matrix has a shape of 3x3 or b x 3x3
            intrinsic_matrix = intrinsic_matrices[0] if len(intrinsic_matrices.shape) == 3 else intrinsic_matrices
            
            model_type = config.MODEL.NAME
            
            if model_type == "FTL":
                b, v = extrinsic_matrices.shape[0:2]
                # pose2d_pred:
                # pose3d_pred:
                heatmaps, pose2d_pred, pose3d_pred = model(imgs, extrinsic_matrices, intrinsic_matrices)
            elif model_type == "alg" or model_type == "ransac":
                # heatmaps: b x v x 21 x 64 x 64
                # pose2d_pred: b x v x 21 x 2 [located in the original image of size 640(W) x 480(H)]
                # pose3d_pred: b x 21 x 3 [world coord]

                pose2d_gt[:,:,:,0] = pose2d_gt[:,:,:,0] * orig_width / heatmap_size[0]
                pose2d_gt[:,:,:,1] = pose2d_gt[:,:,:,1] * orig_height / heatmap_size[1]
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices) # b x v x 3 x 4

                pose3d_pred,\
                pose2d_pred,\
                heatmaps_pred,\
                confidences_pred = model(imgs.cuda(device, non_blocking=True), proj_matrices.cuda(device, non_blocking=True), orig_img_size=val_loader.dataset.orig_img_size)

            elif "vol" in model_type:
                # pose3d_pred (torch.tensor)
                # pose2d_pred (torch.tensor) b x v x 21 x 2
                # heatmaps_pred (torch.tensor)
                # volumes_pred (torch.tensor)
                # confidences_pred (torch.tensor)
                # cuboids_pred (list)
                # coord_volumes_pred (torch.tensor)
                # base_points_pred (torch.tensor)

                # the intrinsic matrix needs to be scaled so that it represents 64 x 64 images
                intrinsic_matrix = update_after_resize(
                    intrinsic_matrix,
                    (orig_height, orig_width),
                    tuple(heatmap_size))
                proj_matrices = (intrinsic_matrix @ extrinsic_matrices) # b x v x 3 x 4
                if config.MODEL.BACKBONE_NAME == 'CPM_volumetric':
                    centermaps = ret['centermaps']
                    heatmaps_gt = ret['heatmaps']

                    pose3d_pred,\
                    pose2d_pred,\
                    heatmaps_pred,\
                    volumes_pred,\
                    confidences_pred,\
                    coord_volumes_pred,\
                    base_points_pred\
                        = model(imgs.cuda(device, non_blocking=True), centermaps.cuda(device, non_blocking=True), proj_matrices.cuda(device, non_blocking=True))

                else:
                    pose3d_pred,\
                    pose2d_pred,\
                    heatmaps_pred,\
                    volumes_pred,\
                    confidences_pred,\
                    coord_volumes_pred,\
                    base_points_pred\
                        = model(imgs.cuda(device, non_blocking=True), proj_matrices.cuda(device, non_blocking=True))

            batch_size, n_views= imgs.shape[0], imgs.shape[1]
            n_joints = pose3d_pred.shape[1]
            #pose3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)
            scale_keypoints_3d = config.MODEL.SCALE_KEYPOINTS_3D if hasattr(config.MODEL, "SCALE_KEYPOINTS_3D") else 1.0

            # 1-view case
            if n_views == 1:
                base_joint = 9 # the middle finger root is specified as the cuboid center

                keypoints_3d_gt_transformed = pose3d_gt.clone()
                keypoints_3d_gt_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_gt_transformed[:, base_joint:base_joint + 1]
                pose3d_gt = keypoints_3d_gt_transformed

                keypoints_3d_pred_transformed = pose3d_pred.clone()
                keypoints_3d_pred_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_pred_transformed[:, base_joint:base_joint + 1]
                pose3d_pred = keypoints_3d_pred_transformed

            # calculate losses
            if model_type == "FTL":
                # pose3d_gt_rel = torch.zeros(pose3d_gt.shape).float()
                # pose3d_gt_rel[:,1:,:] = pose3d_gt[:,1:,:] - pose3d_gt[:,0:1,:]
                # pose3d_gt_pred = torch.zeros(pose3d_pred.shape).float()
                # pose3d_gt_pred[:,1:,:] = pose3d_pred[:,1:,:] - pose3d_pred[:,0:1,:]
                
                loss_dict = recorder.computeLosses(
                    pose2d_pred=pose2d_pred.view(-1,21,2),
                    pose2d_gt=pose2d_gt.view(-1,21,2).cuda(device),
                    pose3d_pred=pose3d_pred,
                    pose3d_gt=pose3d_gt.cuda(device))
            else:
                if False:
                    # pose3d b x 21 x 3
                    pose2d_pred_reproj = torch.cat([project3Dto2D(pose3d_pred[:,k], proj_matrices).unsqueeze(2) for k in range(pose3d_pred.shape[1])], dim=2)# 
                
                item_dict = {
                    'pose3d_pred': pose3d_pred, # b x 21 x 3
                    'pose3d_gt': pose3d_gt,
                    'pose3d_binary_validity_gt': torch.ones((batch_size, n_joints, 1), dtype=torch.float32, device=device),  # 源代码中该参数被设置为全1.0的张量，见human36m.py的172行
                }
                if 'vol' in model_type:
                    item_dict['coord_volumes_pred'] = coord_volumes_pred
                    item_dict['volumes_pred'] = volumes_pred
                if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                    item_dict['data_idx'] = ret['data_idx'] # (b,)
                if config.LOSS.WITH_HEATMAP_LOSS:
                    item_dict['heatmaps_gt'] = heatmaps_gt.cuda(device, non_blocking=True)
                    item_dict['heatmaps_pred'] = heatmaps_pred
                if config.LOSS.WITH_KCS_LOSS:
                    global KC_matrix
                    KC_matrix = KC_matrix.cuda(device, non_blocking=True) # 20 x 21
                    kinematic_chain_gt = KC_matrix @ pose3d_gt # b x 20 x 3

                    kinematic_chain_pred = KC_matrix @ pose3d_pred # b x 20 x 3
                    kinematic_chain_Pred_T = kinematic_chain_pred.clone().transpose(1,2) # b x 3 x 20

                    item_dict['KCS_gt'] =  kinematic_chain_gt @ kinematic_chain_gt.transpose(1,2) # b x 20 x 20
                    item_dict['KCS_pred'] = kinematic_chain_pred @ kinematic_chain_Pred_T
                    if config.LOSS.WITH_KCS_TC_LOSS:
                        item_dict['data_idx'] = ret['data_idx'] # (b,)
                if config.LOSS.WITH_POSE2D_LOSS:
                    item_dict['pose2d_pred'] = pose2d_pred.view(-1,21,2)
                    item_dict['pose2d_gt'] = pose2d_gt.view(-1,21,2).cuda(device, non_blocking=True)
                    item_dict['pose2d_visibility'] = visibility.view(-1,21).cuda(device, non_blocking=True)

                loss_dict = recorder.computeLosses(item_dict)

            total_loss = loss_dict['total_loss']
            heatmap_loss = loss_dict['heatmap_loss']
            pose2d_loss = loss_dict['pose2d_loss']
            pose3d_loss = loss_dict['pose3d_loss'] # normal value: 100-200
            volumetric_ce_loss = loss_dict['volumetric_ce_loss']
            KCS_loss = loss_dict['KCS_loss']
            KCS_TC_loss = loss_dict['KCS_TC_loss']
            time_consistency_loss = loss_dict['time_consistency_loss']

            if master and i % config.PRINT_FREQ == 0:
                recorder.computeAvgLosses()
                # measure elapsed time
                batch_time = (time.time() - end) / config.PRINT_FREQ
                end = time.time()

                msg = 'Test: [{0}/{1}]\t' \
                'Time {batch_time:.3f}s\t' \
                'Speed {speed:.1f} samples/s\t' \
                'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})\t'.format(
                    i+1, len(val_loader), batch_time=batch_time,
                    speed=imgs.shape[0]/batch_time,
                    total_loss=total_loss.item(),
                    total_loss_avg=recorder.avg_total_loss)

                if heatmap_loss:
                    msg += '\tHeatmapLoss {heatmap_loss:.5f} ({heatmap_loss_avg:.5f})'.format(
                        heatmap_loss = heatmap_loss.item(), heatmap_loss_avg = recorder.avg_heatmap_loss
                    )

                if pose2d_loss:
                    msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                        pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
                    )

                if pose3d_loss:
                    msg += '\tPose3DLoss {pose3d_loss:.5f} ({pose3d_loss_avg:.5f})'.format(
                        pose3d_loss = pose3d_loss.item(), pose3d_loss_avg = recorder.avg_pose3d_loss
                    )
                if KCS_loss:
                    msg += '\tKCS_loss {KCS_loss:.5f} ({KCS_loss_avg:.5f})'.format(
                        KCS_loss = KCS_loss.item(), KCS_loss_avg = recorder.avg_KCS_loss
                    )
                if KCS_TC_loss:
                    msg += '\tKCS_TC_loss {KCS_TC_loss:.5f} ({KCS_TC_loss_avg:.5f})'.format(
                        KCS_TC_loss = KCS_TC_loss.item(), KCS_TC_loss_avg = recorder.avg_KCS_TC_loss
                    )
                if volumetric_ce_loss:
                    msg += '\tVolumetricCELoss {volumetric_ce_loss:.5f} ({volumetric_ce_loss_avg:.5f})'.format(
                        volumetric_ce_loss = volumetric_ce_loss.item(), volumetric_ce_loss_avg = recorder.avg_volumetric_ce_loss
                    )
                if time_consistency_loss:
                    msg += '\tTime_consistency_loss {time_consistency_loss:.5f} ({time_consistency_loss_avg:.5f})'.format(
                        time_consistency_loss = time_consistency_loss.item(), time_consistency_loss_avg = recorder.avg_time_consistency_loss
                    )

                logger.info(msg)

            if debug and i%3 == 0: break

        recorder.computeAvgLosses()
        
        if master:
            global_steps = writer_dict['valid_global_steps']
            if config.LOSS.WITH_HEATMAP_LOSS:
                writer.add_scalar('val_loss/heatmap_loss', recorder.avg_heatmap_loss, global_steps)
            if config.LOSS.WITH_POSE2D_LOSS:
                writer.add_scalar('val_loss/pose2d_loss', recorder.avg_pose2d_loss, global_steps)
            if config.LOSS.WITH_POSE3D_LOSS:
                writer.add_scalar('val_loss/pose3d_loss', recorder.avg_pose3d_loss, global_steps)
            if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
            if config.LOSS.WITH_KCS_LOSS:
                writer.add_scalar('val_loss/KCS_loss', recorder.avg_KCS_loss, global_steps)
            if config.LOSS.WITH_KCS_TC_LOSS:
                writer.add_scalar('val_loss/KCS_TC_loss', recorder.avg_KCS_TC_loss, global_steps)
            if config.LOSS.WITH_BONE_LOSS:
                writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
            if config.LOSS.WITH_JOINTANGLE_LOSS:
                writer.add_scalar('val_loss/jointangle_loss', recorder.avg_jointangle_loss, global_steps)

            writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)                 
        
        writer_dict['valid_global_steps'] += 1

    return recorder

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, config, criterion):
        self.config = config
        self.criterion = criterion

        self.total_loss = 0.
        self.heatmap_loss = 0. if config.LOSS.WITH_HEATMAP_LOSS else None
        self.pose2d_loss = 0. if config.LOSS.WITH_POSE2D_LOSS else None
        self.pose3d_loss = 0. if config.LOSS.WITH_POSE3D_LOSS else None
        self.volumetric_ce_loss = 0. if config.LOSS.WITH_VOLUMETRIC_CE_LOSS else None
        self.time_consistency_loss = 0. if config.LOSS.WITH_TIME_CONSISTENCY_LOSS else None
        self.bone_loss = 0. if config.LOSS.WITH_BONE_LOSS else None
        self.jointangle_loss = 0. if config.LOSS.WITH_JOINTANGLE_LOSS else None
        self.KCS_loss = 0. if config.LOSS.WITH_KCS_LOSS else None
        self.KCS_TC_loss = 0. if config.LOSS.WITH_KCS_TC_LOSS else None
        self.n = 0
    
    def computeAvgLosses(self):
        iter_count = self.n
        self.avg_total_loss = self.total_loss / iter_count

        loss_dict = {'total_loss': self.avg_total_loss}

        if self.heatmap_loss:
            self.avg_heatmap_loss = self.heatmap_loss / iter_count
            loss_dict['heatmap_loss'] = self.avg_heatmap_loss

        if self.pose2d_loss:
            self.avg_pose2d_loss = self.pose2d_loss / iter_count
            loss_dict['pose2d_loss'] = self.avg_pose2d_loss

        if self.pose3d_loss:
            self.avg_pose3d_loss = self.pose3d_loss / iter_count
            loss_dict['pose3d_loss'] = self.avg_pose3d_loss
        
        if self.volumetric_ce_loss:
            self.avg_volumetric_ce_loss = self.volumetric_ce_loss / iter_count
            loss_dict['volumetric_ce_loss'] = self.avg_volumetric_ce_loss

        if self.time_consistency_loss:
            self.avg_time_consistency_loss = self.time_consistency_loss / iter_count
            loss_dict['time_consistency_loss'] = self.avg_time_consistency_loss

        if self.bone_loss:
            self.avg_bone_loss = self.bone_loss / iter_count
            loss_dict['bone_loss'] = self.avg_bone_loss

        if self.jointangle_loss:
            self.avg_jointangle_loss = self.jointangle_loss / iter_count
            loss_dict['jointangle_loss'] = self.avg_jointangle_loss

        if self.KCS_loss:
            self.avg_KCS_loss = self.KCS_loss / iter_count
            loss_dict['KCS_loss'] = self.avg_KCS_loss
        
        if self.KCS_TC_loss:
            self.avg_KCS_TC_loss = self.KCS_TC_loss / iter_count
            loss_dict['KCS_TC_loss'] = self.avg_KCS_TC_loss
        
        return loss_dict


    def computeLosses(self, item_dict, n=1):
        """
        pose2d_pred=None
        pose2d_gt=None
        pose2d_visibility b x 21
        pose3d_pred b x 21 x 3
        pose3d_gt=None
        pose3d_binary_validity_gt=None
        coord_volumes_pred=None
        volumes_pred=None,
        """
        self.n += n
        items = item_dict.keys()
        loss_dict = {
            'heatmap_loss': None,
            'pose2d_loss': None,
            'pose3d_loss': None,
            'volumetric_ce_loss': None,
            'time_consistency_loss': None,
            'jointangle_loss': None,
            'bone_loss': None,
            'KCS_loss': None,
            'KCS_TC_loss': None,
            'total_loss': 0
            }

        total_loss = 0
        loss_functions = self.criterion.keys()

        if 'heatmap_loss' in loss_functions:
            heatmap_loss = self.criterion['heatmap_loss'](item_dict['heatmaps_pred'], item_dict['heatmaps_gt'])
            self.heatmap_loss += heatmap_loss.item()
            total_loss += self.config.LOSS.HEATMAP_LOSS_FACTOR * heatmap_loss
            loss_dict['heatmap_loss'] = heatmap_loss
        
        if 'pose2d_loss' in loss_functions and 'pose2d_pred' in items:
            pose2d_loss = self.criterion['pose2d_loss'](item_dict['pose2d_pred'], item_dict['pose2d_gt'], visibility=item_dict['pose2d_visibility'])
            self.pose2d_loss += pose2d_loss.item()
            total_loss += self.config.LOSS.POSE2D_LOSS_FACTOR * pose2d_loss
            loss_dict['pose2d_loss'] = pose2d_loss

        if 'pose3d_loss' in loss_functions and 'pose3d_pred' in items:
            pose3d_loss = self.criterion['pose3d_loss'](item_dict['pose3d_pred'], item_dict['pose3d_gt'])
            self.pose3d_loss += pose3d_loss.item()
            total_loss += self.config.LOSS.POSE3D_LOSS_FACTOR * pose3d_loss
            loss_dict['pose3d_loss'] = pose3d_loss

        if 'volumetric_ce_loss' in loss_functions and 'coord_volumes_pred' in items:
            volumetric_ce_loss = self.criterion['volumetric_ce_loss'](item_dict['coord_volumes_pred'], item_dict['volumes_pred'], item_dict['pose3d_gt'], item_dict['pose3d_binary_validity_gt'])
            self.volumetric_ce_loss += volumetric_ce_loss.item()
            total_loss += self.config.LOSS.VOLUMETRIC_LOSS_FACTOR * volumetric_ce_loss
            loss_dict['volumetric_ce_loss'] = volumetric_ce_loss
        if 'time_consistency_loss' in loss_functions and 'pose3d_pred' in items:
            data_idx, pose3d_pred, pose3d_gt = item_dict['data_idx'], item_dict['pose3d_pred'], item_dict['pose3d_gt']
            split_idx, time_consistency_loss = 1, torch.tensor(0., device=pose3d_pred.device)

            while split_idx < data_idx.shape[0]:
                if data_idx[split_idx] != data_idx[0]: break
                split_idx += 1

            if split_idx >= 2: 
                pose3d_pred_last_frame = pose3d_pred[0:split_idx-1].clone()
                pose3d_gt_last_frame = pose3d_gt[0:split_idx-1]

                time_consistency_loss += self.criterion['time_consistency_loss'](
                    pose3d_pred[1:split_idx] - pose3d_pred_last_frame,
                    pose3d_gt[1:split_idx] - pose3d_gt_last_frame)
            
            if data_idx.shape[0] - split_idx >= 2:
                pose3d_pred_last_frame = pose3d_pred[split_idx:-1].clone()
                pose3d_gt_last_frame = pose3d_gt[split_idx:-1]

                time_consistency_loss += self.criterion['time_consistency_loss'](
                    pose3d_pred[split_idx+1:] - pose3d_pred_last_frame,
                    pose3d_gt[split_idx+1:] - pose3d_gt_last_frame)

            self.time_consistency_loss += time_consistency_loss.item()
            total_loss += self.config.LOSS.TIME_CONSISTENCY_LOSS_FACTOR * time_consistency_loss
            loss_dict['time_consistency_loss'] = time_consistency_loss
        
        if 'bone_loss' in loss_functions or 'jointangle_loss' in loss_functions and pose2d_pred is not None:
            pose2d_rel_gt = scale_pose2d(pose2d_gt)
            pose2d_rel_pred = scale_pose2d(pose2d_pred)

            if 'bone_loss' in loss_functions:              
                bone_loss = self.criterion['bone_loss'](pose2d_rel_pred[:,:,0:2], pose2d_rel_gt[:,:,0:2])
                self.bone_loss += bone_loss.item()
                total_loss += self.config.LOSS.BONE_LOSS_FACTOR * bone_loss
                loss_dict['bone_loss'] = bone_loss

            if 'jointangle_loss' in loss_functions:
                # append a z-axis value to each uv coord for the requirement of torch.cross
                zeros = torch.zeros((pose2d_rel_gt.shape[0], 21, 1), dtype=pose2d_pred.dtype)
                if pose2d_pred.device.type == 'cuda':
                    zeros = zeros.cuda()

                pose2d_rel_gt_with_z = torch.cat(
                    (pose2d_rel_gt[:,:,0:2], zeros),
                    dim=2)
                pose2d_rel_pred_with_z = torch.cat(
                    (pose2d_rel_pred[:,:,0:2], zeros),
                    dim=2)

                jointangle_loss = self.criterion['jointangle_loss'](pose2d_rel_pred_with_z, pose2d_rel_gt_with_z)
                self.jointangle_loss += jointangle_loss.item()
                total_loss += self.config.LOSS.JOINTANGLE_LOSS_FACTOR * jointangle_loss
                loss_dict['jointangle_loss'] = jointangle_loss

        if 'KCS_loss' in loss_functions and 'KCS_pred' in items:
            KCS_gt, KCS_pred = item_dict['KCS_gt'], item_dict['KCS_pred']
            KCS_loss = self.criterion['KCS_loss'](KCS_gt, KCS_pred)
            self.KCS_loss += KCS_loss.item()
            total_loss += self.config.LOSS.KCS_LOSS_FACTOR * KCS_loss
            loss_dict['KCS_loss'] = KCS_loss

            if 'data_idx' in items:
                data_idx = item_dict['data_idx']
                split_idx, KCS_TC_loss = 1, torch.tensor(0., device=KCS_pred.device)

                while split_idx < data_idx.shape[0]:
                    if data_idx[split_idx] != data_idx[0]:
                        break
                    split_idx += 1

                if split_idx >= 2: 
                    KCS_pred_last_frame = KCS_pred[0:split_idx-1].clone()
                    KCS_gt_last_frame = KCS_gt[0:split_idx-1]

                    KCS_TC_loss += self.criterion['KCS_loss'](
                        KCS_pred[1:split_idx] - KCS_pred_last_frame,
                        KCS_gt[1:split_idx] - KCS_gt_last_frame)
        
                if data_idx.shape[0] - split_idx >= 2:
                    KCS_pred_last_frame = KCS_pred[split_idx:-1].clone()
                    KCS_gt_last_frame = KCS_gt[split_idx:-1]

                    KCS_TC_loss += self.criterion['KCS_loss'](
                        KCS_pred[split_idx+1:] - KCS_pred_last_frame,
                        KCS_gt[split_idx+1:] - KCS_gt_last_frame)

                self.KCS_TC_loss += KCS_TC_loss.item()
                total_loss += self.config.LOSS.KCS_TC_LOSS_FACTOR * KCS_TC_loss
                loss_dict['KCS_TC_loss'] = KCS_TC_loss

        self.total_loss += total_loss.item()
        loss_dict['total_loss'] = total_loss

        return loss_dict 


def test_sample(img, pose2d):
    """
    img: torch.tensor of size B x H x W x 3
    pose2d: torch.tensor of size B x 21 x 3
    """
    img_np = cv2.cvtColor(img[0].detach().cpu().numpy().transpose(1,2,0), cv2.COLOR_RGB2BGR)
    plt.figure()
    plt.imshow(img_np)
    for i in range(pose2d[0].shape[0]):
        plt.plot(4*pose2d[0][i][0], 4*pose2d[0][i][1],'r*')
    plt.show()
        