# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

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
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back, scale_pose3d, scale_pose2d
from utils.vis import save_debug_images
from heatmap_decoding import get_final_preds

logger = logging.getLogger(__name__)

def train_helper(epoch, i, args, config, img, model, optimizer, train_loader, writer_dict, output_dir, tb_log_dir,
                heatmaps_gt=None, pose2d_gt=None, pose3d_gt=None, recorder=None,
                fp16=False):
    end = time.time()
    heatmaps_pred, temperature = model(img.cuda())

    pose2d_pred = get_final_preds(heatmaps_pred, config)
    heatmaps_gt = heatmaps_gt.cuda(non_blocking=True)
    pose2d_gt = pose2d_gt.cuda(non_blocking=True)

    # calculate losses
    loss_dict = recorder.computeLosses(
        heatmaps_pred,
        heatmaps_gt,
        pose2d_pred,
        pose2d_gt,
        with_visibility=True
    )
    
    total_loss = loss_dict['total_loss']
    heatmap_loss = loss_dict['heatmap_loss']
    pose2d_loss = loss_dict['pose2d_loss']
    TC_loss = loss_dict['TC_loss']
    bone_loss = loss_dict['bone_loss']

    # compute gradient and do update step
    optimizer.zero_grad()
    if fp16:
        optimizer.backward(total_loss)
    else:
        total_loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time = time.time() - end

    if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
        if i % config.PRINT_FREQ == 0:
            recorder.computeAvgLosses()

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time:.3f}s\t' \
                'Speed {speed:.1f} samples/s\t' \
                'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})'.format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time,
                    speed=img.size(0)/batch_time,
                    total_loss=total_loss.item(),
                    total_loss_avg=recorder.avg_total_loss)
            
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']

            if heatmap_loss is not None:
                msg += '\tHeatmapLoss {heatmap_loss:.5f} ({heatmap_loss_avg:.5f})'.format(
                    heatmap_loss = heatmap_loss.item(), heatmap_loss_avg = recorder.avg_heatmap_loss
                )
                writer.add_scalar('train_loss/heatmap_loss', heatmap_loss, global_steps)
            if pose2d_loss:
                msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                    pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
                )
                writer.add_scalar('train_loss/pose2d_loss', pose2d_loss, global_steps)
            if TC_loss:
                msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                    TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
                )
                writer.add_scalar('train_loss/time_consistency_loss', TC_loss, global_steps)
            if bone_loss:
                msg += '\tBoneLoss {Bone_loss:.5f} ({Bone_loss_avg:.5f})'.format(
                    Bone_loss = bone_loss.item(), bone_loss_avg = recorder.avg_bone_loss
                )
                writer.add_scalar('train_loss/bone_loss', bone_loss, global_steps)
            
            logger.info(msg)
            
            writer.add_scalar('train_loss/total_loss', total_loss, global_steps)
            
            writer.add_scalar('train_loss/trainable_temperature', temperature, global_steps)

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, img, heatmaps_gt, heatmaps_pred, prefix)

    writer_dict['train_global_steps'] += 1

def train(config, args, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, fp16=False):
    """
    - Brief: Training phase
    - params:
        config
        train_loader:
        model:
        criterion: a dict containing loss items
    """
    recorder = AverageMeter(config, criterion)
    writer = writer_dict['writer']
    # switch to train mode
    model.train()

    end = time.time()

    if 'handgraph' in config.DATASET.DATASET.lower():
        for i, (
                img, img_mask, heatmaps_gt, pose2d_gt,
                local_pose3d_gt, mesh2d_gt, local_mesh_pts_gt,
                mesh_tri_idx, cam_proj_mat, img_path
                ) in enumerate(train_loader):
            # img (have been normalized):  size = B x 3 x H x W
            # img_mask (Elements are either 0 or 255): size = B x H x W
            # heatmaps_gt: size = B x 21 x 64(H) x 64(W)
            # pose_2d:  size = B x 21 x 3 (u,v,visibility)
            # local_pose3d_gt:  size = B x 21 x 3 (X, Y, Z)
            # mesh_2d:  size = N x 2 (u,v)
            # local_mesh_pts_gt:    size = N x 3 (X, Y, Z)
            # mesh_tri_idx (The indices of triangles' vertices): size = 1892 x 3
            # cam_proj_mat: [[fx 0  x0]
            #                [0  fy 0 ]
            #                [0  0  1 ]]
            # img_path: a list of the paths of all samples
            # Note: The batch size is determined by img_per_GPU x
            # compute output
            train_helper(epoch, i, config, img, model, optimizer, writer_dict, output_dir, tb_log_dir,
                        heatmaps_gt, pose2d_gt, recorder=recorder, fp16=fp16)

    elif 'rhd' in config.DATASET.DATASET.lower():
        for i, (img, heatmaps_gt, pose2d_gt, img_path) in enumerate(train_loader):
            # img (have been normalized):  size = B x 3 x H x W
            # img_mask (Elements are either 0 or 255): size = B x H x W
            # heatmaps_gt: size = B x 21 x 64(H) x 64(W)
            # pose_2d:  size = B x 21 x 3 (u,v,visibility)
            train_helper(epoch, i, config, img, model, optimizer, writer_dict, output_dir, tb_log_dir,
                        heatmaps_gt, pose2d_gt, recorder=recorder, fp16=fp16)
    
    elif 'frei' in config.DATASET.DATASET.lower():
        for i, (img, heatmaps_gt, pose2d_gt) in enumerate(train_loader):
            # img (have been normalized):  size = B x 3 x H x W
            # heatmaps_gt: size = B x 21 x 64(H) x 64(W)
            # pose_2d:  size = B x 21 x 3 (u,v,visibility)
            train_helper(epoch, i, config, img, model, optimizer, writer_dict, output_dir, tb_log_dir,
                    heatmaps_gt, pose2d_gt, recorder=recorder, fp16=fp16)
    
    elif 'mhp_kpt' == config.DATASET.DATASET.lower():
        for i, (img, heatmaps_gt, pose2d_gt, pose3d_gt) in enumerate(train_loader):
            train_helper(epoch, i, args, config, img, model, optimizer, train_loader, writer_dict, output_dir, tb_log_dir,
                        heatmaps_gt, pose2d_gt, recorder=recorder, fp16=fp16)
            
            #if i==10:break
    
    # only keypoint depths are regressed
    elif False:#'mhp_seq' ==  config.DATASET.DATASET.lower():
        intrin_m = torch.from_numpy(train_loader.dataset.intrinsic_matrix.T).cuda(non_blocking=True)
        center_idx = config.DATASET.N_FRAMES // 2 # 27 // 2 = 13
        # reusable data
        reusable_flag = (config.DATASET.N_FRAMES > config.DATASET.SAMPLE_STRIDE)
        reusable_frames = None
        reusable_pose3d_cam_gt = None
        
        for i, (frames, pose2d_gt, pose3d_cam_gt) in enumerate(train_loader):
            # Length = min(sample_len, sample_stride)
            # frames (have been normalized):  size = Batch x Length x 3 x H x W
            # pose2d_gt:  size = B x Length x 21 x 3 (u,v,visibility)
            # pose3d_gt:  size = B x Length x 21 x 3 (u,v,d) cam coord

            first_flag = (frames.shape[1] == config.DATASET.N_FRAMES)

            # compute output
            #print('train',i,':',frames.shape[1],first_flag, reusable_flag, type(reusable_frames))
            entire_frames = frames if first_flag or not reusable_flag else torch.cat((reusable_frames, frames), dim=1)
            depth_pred_center_frame_rel = model(entire_frames) # b x 20 x 1

            entire_pose3d_cam_gt = pose3d_cam_gt if first_flag or not reusable_flag else torch.cat((reusable_pose3d_cam_gt, pose3d_cam_gt), dim=1) # b x N_frames x 21 x 3
            pose3d_cam_gt_center_frame = entire_pose3d_cam_gt[:,center_idx].cuda(non_blocking=True) # b x 21 x 3

            # depth-relative to the palm
            root_depth = pose3d_cam_gt_center_frame[:,0:1,-1:] # b x 1 x 1
            depth_gt_center_frame_rel = pose3d_cam_gt_center_frame[:,1:,-1:] - root_depth # b x 20 x 1

            if reusable_flag:
                start_idx = config.DATASET.SAMPLE_STRIDE
                reusable_frames = entire_frames[:,start_idx:]
                reusable_pose3d_cam_gt = entire_pose3d_cam_gt[:,start_idx:]

            #continue
            loss_dict = recorder.computeLosses(
                pose3d_pred=depth_pred_center_frame_rel, # b x 20 x 1
                pose3d_gt=depth_gt_center_frame_rel,     # b x 20 x 1
                with_visibility=False
            )

            total_loss=loss_dict['total_loss']
            pose2d_loss = loss_dict['pose2d_loss']
            pose3d_loss = loss_dict['pose3d_loss']
            TC_loss = loss_dict['TC_loss']
            bone_loss = loss_dict['bone_loss']
            jointangle_loss = loss_dict['jointangle_loss']

            # compute gradient and do update step
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(total_loss)
            else:
                total_loss.backward()
            optimizer.step()

            if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
                if i % config.PRINT_FREQ == 0:
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
                            speed=frames.size(0)/batch_time,
                            total_loss=total_loss.item(),
                            total_loss_avg=recorder.avg_total_loss)

                    if pose3d_loss:
                        msg += '\tPose3DLoss {pose3d_loss:.5f} ({pose3d_loss_avg:.5f})'.format(
                            pose3d_loss = pose3d_loss.item(), pose3d_loss_avg = recorder.avg_pose3d_loss
                        )
                        writer.add_scalar('train_loss/pose3d_loss', pose3d_loss, global_steps)
                    if pose2d_loss:
                        msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                            pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
                        )
                        writer.add_scalar('train_loss/pose2d_loss', pose2d_loss, global_steps)
                    if TC_loss:
                        msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                            TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
                        )
                        writer.add_scalar('train_loss/time_consistency_loss', TC_loss, global_steps)
                    if bone_loss:
                        msg += '\tBoneLoss {Bone_loss:.5f} ({Bone_loss_avg:.5f})'.format(
                            Bone_loss = bone_loss.item(), Bone_loss_avg = recorder.avg_bone_loss
                        )
                        writer.add_scalar('train_loss/bone_loss', bone_loss, global_steps)
                    if jointangle_loss:
                        msg += '\tjointangle_loss {jointangle_loss:.5f} ({jointangle_loss_avg:.5f})'.format(
                            jointangle_loss = jointangle_loss.item(), jointangle_loss_avg = recorder.avg_jointangle_loss
                        )
                        writer.add_scalar('train_loss/jointangle_loss', jointangle_loss, global_steps)                 
                    logger.info(msg)
                        
                    writer.add_scalar('train_loss/total_loss', total_loss, global_steps)
            #if i%20 == 0: break

            writer_dict['train_global_steps'] += 1

    # 3D poses (cameara) regressed
    elif 'mhp_seq' ==  config.DATASET.DATASET.lower():
        intrin_m = torch.from_numpy(train_loader.dataset.intrinsic_matrix.T).cuda(non_blocking=True)
        center_idx = config.DATASET.N_FRAMES // 2 # 27 // 2 = 13
        # reusable data
        reusable_flag = (config.DATASET.N_FRAMES > config.DATASET.SAMPLE_STRIDE)
        reusable_frames = None
        reusable_pose2d_gt = None
        reusable_pose3d_cam_gt = None
        
        for i, (frames, pose2d_gt, pose3d_cam_gt) in enumerate(train_loader):
            # Length = min(sample_len, sample_stride)
            # frames (have been normalized):  size = Batch x Length x 3 x H x W
            # pose2d_gt:  size = B x Length x 21 x 3 (u,v,visibility)
            # pose3d_gt:  size = B x Length x 21 x 3 (u,v,d) cam coord
            first_flag = (frames.shape[1] == config.DATASET.N_FRAMES)

            # compute output
            #print('train',i,':',frames.shape[1],first_flag, reusable_flag, type(reusable_frames))
            
            if False: # relative 3D pose prediction (bad performance)
                entire_frames = frames if first_flag or not reusable_flag else torch.cat((reusable_frames, frames), dim=1)
                pose3d_cam_pred_center_frame_rel = model(entire_frames) # b x 21 x 3
                
                # b x N_frames x 21 x 3
                entire_pose3d_cam_gt = pose3d_cam_gt if first_flag or not reusable_flag else torch.cat((reusable_pose3d_cam_gt, pose3d_cam_gt), dim=1)
                # b x 21 x 3
                pose3d_cam_gt_center_frame = entire_pose3d_cam_gt[:,center_idx].cuda(non_blocking=True)
            
                # depth-relative to the palm. size: b x 1 x 1
                root_depth = pose3d_cam_gt_center_frame[:,0:1,-1:]
                # get z values of all keypoints. size b x 21 x 1
                pose3d_cam_gt_center_frame_rel_z = pose3d_cam_gt_center_frame[:,:,-1:].clone() - root_depth
                pose3d_cam_pred_center_frame_z = pose3d_cam_pred_center_frame_rel[:,:,-1:].clone() + root_depth
                # b x 21 x 3
                pose3d_cam_gt_center_frame_rel = torch.cat((pose3d_cam_gt_center_frame[:,:,0:2], pose3d_cam_gt_center_frame_rel_z), dim=2)
                # b x 21 x 3
                pose3d_cam_pred_center_frame = torch.cat((pose3d_cam_pred_center_frame_rel[:,:,0:2], pose3d_cam_pred_center_frame_z), dim=2)
                
                pose2d_reproj_center_frame_xyz = torch.matmul(
                pose3d_cam_pred_center_frame, # b x 21 x 3
                intrin_m # 3 x 3
                ) # get b x 21 x 3 

                pose2d_reproj_center_frame_uv = pose2d_reproj_center_frame_xyz[:,:,0:2] / pose2d_reproj_center_frame_xyz[:,:,2:]
                
                entire_pose2d_gt = pose2d_gt if first_flag or not reusable_flag else torch.cat((reusable_pose2d_gt, pose2d_gt), dim=1) 
                pose2d_gt_center_frame_wo_vis = entire_pose2d_gt[:,center_idx,:,0:2].cuda(non_blocking=True)
                
                if reusable_flag:
                    start_idx = config.DATASET.SAMPLE_STRIDE
                    reusable_frames = entire_frames[:,start_idx:]
                    reusable_pose2d_gt = entire_pose2d_gt[:,start_idx:]
                    reusable_pose3d_cam_gt = entire_pose3d_cam_gt[:,start_idx:]

                #continue
                loss_dict = recorder.computeLosses(
                    pose2d_pred=pose2d_reproj_center_frame_uv, # b x 21 x 2
                    pose2d_gt=pose2d_gt_center_frame_wo_vis,  # b x 21 x 2
                    pose3d_pred=pose3d_cam_pred_center_frame_rel, # b x 21 x 3
                    pose3d_gt=pose3d_cam_gt_center_frame_rel,     # b x 21 x 3
                    with_visibility=False
                )
            else:
                entire_frames = frames if first_flag or not reusable_flag else torch.cat((reusable_frames, frames), dim=1)
                pose3d_cam_pred_center_frame = model(entire_frames) # b x 21 x 3
                # b x N_frames x 21 x 3
                entire_pose3d_cam_gt = pose3d_cam_gt if first_flag or not reusable_flag else torch.cat((reusable_pose3d_cam_gt, pose3d_cam_gt), dim=1)
                # b x 21 x 3
                pose3d_cam_gt_center_frame = entire_pose3d_cam_gt[:,center_idx].cuda(non_blocking=True)
            
                pose2d_reproj_center_frame_xyz = torch.matmul(
                    pose3d_cam_pred_center_frame, # b x 21 x 3
                    intrin_m # 3 x 3
                    ) # get b x 21 x 3 

                pose2d_reproj_center_frame_uv = pose2d_reproj_center_frame_xyz[:,:,0:2] / pose2d_reproj_center_frame_xyz[:,:,2:]
                
                entire_pose2d_gt = pose2d_gt if first_flag or not reusable_flag else torch.cat((reusable_pose2d_gt, pose2d_gt), dim=1) 
                pose2d_gt_center_frame_wo_vis = entire_pose2d_gt[:,center_idx,:,0:2].cuda(non_blocking=True)
                
                if reusable_flag:
                    start_idx = config.DATASET.SAMPLE_STRIDE
                    reusable_frames = entire_frames[:,start_idx:]
                    reusable_pose2d_gt = entire_pose2d_gt[:,start_idx:]
                    reusable_pose3d_cam_gt = entire_pose3d_cam_gt[:,start_idx:]

                #continue
                loss_dict = recorder.computeLosses(
                    pose2d_pred=pose2d_reproj_center_frame_uv, # b x 21 x 2
                    pose2d_gt=pose2d_gt_center_frame_wo_vis,  # b x 21 x 2
                    pose3d_pred=pose3d_cam_pred_center_frame, # b x 21 x 3
                    pose3d_gt=pose3d_cam_gt_center_frame,     # b x 21 x 3
                    with_visibility=False
                )


            total_loss=loss_dict['total_loss']
            pose2d_loss = loss_dict['pose2d_loss']
            pose3d_loss = loss_dict['pose3d_loss']
            TC_loss = loss_dict['TC_loss']
            bone_loss = loss_dict['bone_loss']
            jointangle_loss = loss_dict['jointangle_loss']

            # compute gradient and do update step
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(total_loss)
            else:
                total_loss.backward()
            optimizer.step()

            if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
                if i % config.PRINT_FREQ == 0:
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
                            speed=frames.size(0)/batch_time,
                            total_loss=total_loss.item(),
                            total_loss_avg=recorder.avg_total_loss)

                    if pose3d_loss:
                        msg += '\tPose3DLoss {pose3d_loss:.5f} ({pose3d_loss_avg:.5f})'.format(
                            pose3d_loss = pose3d_loss.item(), pose3d_loss_avg = recorder.avg_pose3d_loss
                        )
                        writer.add_scalar('train_loss/pose3d_loss', pose3d_loss, global_steps)
                    if pose2d_loss:
                        msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                            pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
                        )
                        writer.add_scalar('train_loss/pose2d_loss', pose2d_loss, global_steps)
                    if TC_loss:
                        msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                            TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
                        )
                        writer.add_scalar('train_loss/time_consistency_loss', TC_loss, global_steps)
                    if bone_loss:
                        msg += '\tBoneLoss {Bone_loss:.5f} ({Bone_loss_avg:.5f})'.format(
                            Bone_loss = bone_loss.item(), Bone_loss_avg = recorder.avg_bone_loss
                        )
                        writer.add_scalar('train_loss/bone_loss', bone_loss, global_steps)
                    if jointangle_loss:
                        msg += '\tjointangle_loss {jointangle_loss:.5f} ({jointangle_loss_avg:.5f})'.format(
                            jointangle_loss = jointangle_loss.item(), jointangle_loss_avg = recorder.avg_jointangle_loss
                        )
                        writer.add_scalar('train_loss/jointangle_loss', jointangle_loss, global_steps)                 
                    logger.info(msg)
                        
                    writer.add_scalar('train_loss/total_loss', total_loss, global_steps)
            #if i%20 == 0: break

            writer_dict['train_global_steps'] += 1

    elif 'fha' in config.DATASET.DATASET.lower():
        for i, (imgs, pose2d_gt, pose3d_gt) in enumerate(train_loader):
            # imgs (have been normalized):  size = Batch x Length x 3 x H x W
            # pose2d_gt:  size = B x Length x 21 x 3 (u,v,visibility)
            # pose3d_gt:  size = B x Length x 21 x 3 (u,v,d) cam coord

            # compute output
            pose3d_pred = model(imgs.cuda(non_blocking=True))
            pose3d_pred = scale_pose3d(pose3d_pred)

            pose3d_gt = torch.squeeze(pose3d_gt[:,-1,:,:], dim=1).cuda(non_blocking=True)
            pose3d_gt = scale_pose3d(pose3d_gt)

            # calculate losses
            loss_dict = recorder.computeLosses(
                pose3d_pred,
                pose3d_gt
            )

            
            total_loss=loss_dict['total_loss']
            pose3d_loss = loss_dict['pose3d_loss']
            pose2d_loss = loss_dict['pose2d_loss']
            TC_loss = loss_dict['TC_loss']
            bone_loss = loss_dict['bone_loss']
            jointangle_loss = loss_dict['jointangle_loss']

            # compute gradient and do update step
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(total_loss)
            else:
                total_loss.backward()
            optimizer.step()
        
            if i % config.PRINT_FREQ == 0:
                recorder.computeAvgLosses_for_RNN()
                # measure elapsed time
                batch_time = (time.time() - end) / config.PRINT_FREQ
                end = time.time()

                if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
                    global_steps = writer_dict['train_global_steps']

                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Lr: {learning_rate:.2e}\t' \
                        'Time {batch_time:.3f}s\t' \
                        'Speed {speed:.1f} samples/s\t' \
                        'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})\t'.format(
                            epoch, i+1, len(train_loader), learning_rate=model.optimizer.state_dict()['param_groups'][0]['lr'],
                            batch_time=batch_time,
                            speed=imgs.size(0)/batch_time,
                            total_loss=total_loss.item(),
                            total_loss_avg=recorder.avg_total_loss)

                    if pose3d_loss:
                        msg += '\tPose3DLoss {pose3d_loss:.5f} ({pose3d_loss_avg:.5f})'.format(
                            pose3d_loss = pose3d_loss.item(), pose3d_loss_avg = recorder.avg_pose3d_loss
                        )
                        writer.add_scalar('train_loss/pose3d_loss', pose3d_loss, global_steps)
                    if TC_loss:
                        msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                            TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
                        )
                        writer.add_scalar('train_loss/time_consistency_loss', TC_loss, global_steps)
                    if bone_loss:
                        msg += '\tBoneLoss {Bone_loss:.5f} ({Bone_loss_avg:.5f})'.format(
                            Bone_loss = bone_loss.item(), Bone_loss_avg = recorder.avg_bone_loss
                        )
                        writer.add_scalar('train_loss/bone_loss', bone_loss, global_steps)
                    if jointangle_loss:
                        msg += '\tjointangle_loss {jointangle_loss:.5f} ({jointangle_loss_avg:.5f})'.format(
                            jointangle_loss = jointangle_loss.item(), jointangle_loss_avg = recorder.avg_jointangle_loss
                        )
                        writer.add_scalar('train_loss/jointangle_loss', jointangle_loss, global_steps)                 
                    logger.info(msg)
                        
                    writer.add_scalar('train_loss/total_loss', total_loss, global_steps)
            #if i%20 == 0: break

            writer_dict['train_global_steps'] += 1

    else:
        print('[Error in funtion.py] Invalid dataset!')
        exit()
    return recorder


def val_helper(i, config, args, img, model, val_loader, output_dir, tb_log_dir,\
                heatmaps_gt=None, pose2d_gt=None, pose3d_gt=None, recorder=None):
    end = time.time()
    # compute output
    heatmaps_pred, _ = model(img.cuda())

    if config.TEST.FLIP_TEST:
        # this part is ugly, because pytorch has not supported negative index
        # input_flipped = model(images[:, :, :, ::-1])
        input_flipped = np.flip(img.cpu().numpy(), 3).copy()
        input_flipped = torch.from_numpy(input_flipped).cuda()
        heatmaps_pred_flipped = model(input_flipped.cuda().float())

        if isinstance(heatmaps_pred_flipped, list):
            heatmaps_pred_flipped = heatmaps_pred_flipped[-1]

        heatmaps_pred_flipped = flip_back(heatmaps_pred_flipped.cpu().numpy(),
                                val_dataset.flip_pairs)
        heatmaps_pred_flipped = torch.from_numpy(heatmaps_pred_flipped.copy()).cuda()


        # feature is not aligned, shift flipped heatmap for higher accuracy
        if config.TEST.SHIFT_HEATMAP:
            heatmaps_pred_flipped[:, :, :, 1:] = \
                heatmaps_pred_flipped.clone()[:, :, :, 0:-1]

        heatmaps_pred = (heatmaps_pred + heatmaps_pred_flipped) * 0.5

    heatmaps_gt = heatmaps_gt.cuda(non_blocking=True)
    pose2d_pred = get_final_preds(heatmaps_pred, config)
    pose2d_gt = pose2d_gt.cuda(non_blocking=True)

    # calculate losses
    loss_dict = recorder.computeLosses(
        heatmaps_pred,
        heatmaps_gt,
        pose2d_pred,
        pose2d_gt,
        with_visibility=True
    )

    total_loss=loss_dict['total_loss']
    heatmap_loss=loss_dict['heatmap_loss']
    pose2d_loss = loss_dict['pose2d_loss']
    TC_loss = loss_dict['TC_loss']
    bone_loss = loss_dict['bone_loss']
    jointangle_loss = loss_dict['jointangle_loss']

    if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
        # measure elapsed time
        batch_time = time.time() - end
        
        if i % config.PRINT_FREQ == 0:
            recorder.computeAvgLosses()

            msg = 'Test: [{0}/{1}]\t' \
            'Time {batch_time:.3f}s\t' \
            'Speed {speed:.1f} samples/s\t' \
            'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})'.format(
                i, len(val_loader), batch_time=batch_time,
                speed=img.size(0)/batch_time,
                total_loss=total_loss.item(),
                total_loss_avg=recorder.avg_total_loss)

            if heatmap_loss is not None:
                msg += '\tHeatmapLoss {heatmap_loss:.5f} ({heatmap_loss_avg:.5f})'.format(
                    heatmap_loss = heatmap_loss.item(), heatmap_loss_avg = recorder.avg_heatmap_loss
                )

            if pose2d_loss is not None:
                msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                    pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
                )
                
            if TC_loss is not None:
                msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                    TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
                )
            if bone_loss is not None:
                msg += '\tBoneLoss {bone_loss:.5f} ({bone_loss_avg:.5f})'.format(
                    bone_loss = bone_loss.item(), bone_loss_avg = recorder.avg_bone_loss
                )
            if jointangle_loss is not None:
                msg += '\tJointAngleLoss {jointangle_loss:.5f} ({jointangle_loss_avg:.5f})'.format(
                    jointangle_loss = jointangle_loss.item(),jointangle_loss_avg = recorder.avg_jointangle_loss
                )
            
            logger.info(msg)

            prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
            )
            save_debug_images(config, img, heatmaps_gt, heatmaps_pred, prefix)

def validate(config, args, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    recorder = AverageMeter(config, criterion)
    writer = writer_dict['writer']
    # switch to evaluate mode
    model.eval()

    if 'handgraph' in config.DATASET.DATASET.lower():
        with torch.no_grad():
            end = time.time()
            for i, (
                img, img_mask, heatmaps_gt, pose2d_gt,
                local_pose3d_gt, mesh2d_gt, local_mesh_pts_gt,
                mesh_tri_idx, cam_proj_mat, img_path) in enumerate(val_loader):
                
                val_helper(config, img, model, optimizer, output_dir, tb_log_dir,
                        heatmaps_gt, pose2d_gt, recorder=recorder)
            recorder.computeAvgLosses()
            global_steps = writer_dict['valid_global_steps']

            writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)
            writer.add_scalar('val_loss/heatmap_loss', recorder.avg_heatmap_loss, global_steps)
            if config.LOSS.WITH_POSE2D_LOSS:
                writer.add_scalar('val_loss/pose2d_loss', recorder.avg_pose2d_loss, global_steps)
            if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
            if config.LOSS.WITH_BONE_LOSS:
                writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
                
            writer_dict['valid_global_steps'] = global_steps + 1
    
    elif 'rhd' in config.DATASET.DATASET.lower():
        with torch.no_grad():
            end = time.time()
            for i, (img, heatmaps_gt, pose2d_gt, img_path) in enumerate(val_loader):
                val_helper(config, img, model, optimizer, output_dir, tb_log_dir,
                        heatmaps_gt, pose2d_gt, recorder=recorder)
                
            recorder.computeAvgLosses()
            global_steps = writer_dict['valid_global_steps']

            writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)
            writer.add_scalar('val_loss/heatmap_loss', recorder.avg_heatmap_loss, global_steps)
            if config.LOSS.WITH_POSE2D_LOSS:
                writer.add_scalar('val_loss/pose2d_loss', recorder.avg_pose2d_loss, global_steps)
            if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
            if config.LOSS.WITH_BONE_LOSS:
                writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
                
            writer_dict['valid_global_steps'] = global_steps + 1

    elif 'frei' in config.DATASET.DATASET.lower():
        with torch.no_grad():
            end = time.time()
            for i, (img, heatmaps_gt, pose2d_gt) in enumerate(val_loader):
                val_helper(i, config, img, model, optimizer, val_loader, output_dir, tb_log_dir,
                        heatmaps_gt, pose2d_gt, recorder=recorder)
                
            recorder.computeAvgLosses()

            if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
                global_steps = writer_dict['valid_global_steps']

                writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)
                if config.LOSS.WITH_HEATMAP_LOSS:
                    writer.add_scalar('val_loss/heatmap_loss', recorder.avg_heatmap_loss, global_steps)
                if config.LOSS.WITH_POSE2D_LOSS:
                    writer.add_scalar('val_loss/pose2d_loss', recorder.avg_pose2d_loss, global_steps)
                if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                    writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
                if config.LOSS.WITH_BONE_LOSS:
                    writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
                if config.LOSS.WITH_JOINTANGLE_LOSS:
                    writer.add_scalar('val_loss/jointangle_loss', recorder.avg_jointangle_loss, global_steps)
            writer_dict['valid_global_steps'] += 1
    
    elif 'mhp_kpt' == config.DATASET.DATASET.lower():
        with torch.no_grad():
            end = time.time()
            for i, (img, heatmaps_gt, pose2d_gt, pose3d_gt) in enumerate(val_loader):
                val_helper(i, config, args, img, model, val_loader, output_dir, tb_log_dir,
                        heatmaps_gt, pose2d_gt, recorder=recorder)
                # if i == 10:break
            
            recorder.computeAvgLosses()

            if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
                global_steps = writer_dict['valid_global_steps']

                writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)
                if config.LOSS.WITH_HEATMAP_LOSS:
                    writer.add_scalar('val_loss/heatmap_loss', recorder.avg_heatmap_loss, global_steps)
                if config.LOSS.WITH_POSE2D_LOSS:
                    writer.add_scalar('val_loss/pose2d_loss', recorder.avg_pose2d_loss, global_steps)
                if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                    writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
                if config.LOSS.WITH_BONE_LOSS:
                    writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
                if config.LOSS.WITH_JOINTANGLE_LOSS:
                    writer.add_scalar('val_loss/jointangle_loss', recorder.avg_jointangle_loss, global_steps)
            writer_dict['valid_global_steps'] += 1

    elif False:#'mhp_seq' == config.DATASET.DATASET.lower():
        if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
            with torch.no_grad():
                end = time.time()
                center_idx = config.DATASET.N_FRAMES // 2 # 27 // 2 = 13
                intrin_m = torch.from_numpy(val_loader.dataset.intrinsic_matrix.T).cuda(non_blocking=True)
                
                reusable_flag = (config.DATASET.N_FRAMES > config.DATASET.SAMPLE_STRIDE)
                reusable_frames = None
                reusable_pose3d_cam_gt = None
                
                for i, (frames, pose2d_gt, pose3d_cam_gt) in enumerate(train_loader):
                    # Length = min(sample_len, sample_stride)
                    # frames (have been normalized):  size = Batch x Length x 3 x H x W
                    # pose2d_gt:  size = B x Length x 21 x 3 (u,v,visibility)
                    # pose3d_gt:  size = B x Length x 21 x 3 (u,v,d) cam coord

                    first_flag = (frames.shape[1] == config.DATASET.N_FRAMES)

                    # compute output
                    #print('train',i,':',frames.shape[1],first_flag, reusable_flag, type(reusable_frames))
                    entire_frames = frames if first_flag or not reusable_flag else torch.cat((reusable_frames, frames), dim=1)
                    depth_pred_center_frame_rel = model(entire_frames) # b x 20 x 1

                    entire_pose3d_cam_gt = pose3d_cam_gt if first_flag or not reusable_flag else torch.cat((reusable_pose3d_cam_gt, pose3d_cam_gt), dim=1) # b x N_frames x 21 x 3
                    pose3d_cam_gt_center_frame = entire_pose3d_cam_gt[:,center_idx].cuda(non_blocking=True) # b x 21 x 3

                    # depth-relative to the palm
                    root_depth = pose3d_cam_gt_center_frame[:,0:1,-1:] # b x 1 x 1
                    depth_gt_center_frame_rel = pose3d_cam_gt_center_frame[:,1:,-1:] - root_depth # b x 20 x 1

                    if reusable_flag:
                        start_idx = config.DATASET.SAMPLE_STRIDE
                        reusable_frames = entire_frames[:,start_idx:]
                        reusable_pose3d_cam_gt = entire_pose3d_cam_gt[:,start_idx:]

                    #continue
                    loss_dict = recorder.computeLosses(
                        pose3d_pred=depth_pred_center_frame_rel, # b x 20 x 1
                        pose3d_gt=depth_gt_center_frame_rel,     # b x 20 x 1
                        with_visibility=False
                    )

                    loss_dict = recorder.computeLosses(
                        pose2d_pred=pose2d_reproj_center_frame_uv, # b x 21 x 2
                        pose2d_gt=pose2d_gt_center_frame_wo_vis,  # b x 21 x 2
                        pose3d_pred=pose3d_cam_pred_center_frame_rel, # b x 21 x 3
                        pose3d_gt=pose3d_cam_gt_center_frame_rel,     # b x 21 x 3
                        with_visibility=False
                    )

                    total_loss=loss_dict['total_loss']
                    pose2d_loss = loss_dict['pose2d_loss']
                    pose3d_loss = loss_dict['pose3d_loss']
                    TC_loss = loss_dict['TC_loss']
                    bone_loss = loss_dict['bone_loss']
                    jointangle_loss = loss_dict['jointangle_loss']

                    if i % config.PRINT_FREQ == 0:
                        recorder.computeAvgLosses()
                        # measure elapsed time
                        batch_time = (time.time() - end) / config.PRINT_FREQ
                        end = time.time()

                        msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time:.3f}s\t' \
                        'Speed {speed:.1f} samples/s\t' \
                        'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})\t'.format(
                            i+1, len(val_loader), batch_time=batch_time,
                            speed=frames.size(0)/batch_time,
                            total_loss=total_loss.item(),
                            total_loss_avg=recorder.avg_total_loss)

                        if pose3d_loss:
                            msg += '\tPose3DLoss {pose3d_loss:.5f} ({pose3d_loss_avg:.5f})'.format(
                                pose3d_loss = pose3d_loss.item(), pose3d_loss_avg = recorder.avg_pose3d_loss
                            )

                        if pose2d_loss:
                            msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                                pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
                            )

                        if TC_loss:
                            msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                                TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
                            )
                            
                        if bone_loss:
                            msg += '\tBoneLoss {Bone_loss:.5f} ({Bone_loss_avg:.5f})'.format(
                                Bone_loss = bone_loss.item(), Bone_loss_avg = recorder.avg_bone_loss
                            )
                            
                        if jointangle_loss:
                            msg += '\tJointangle_loss {jointangle_loss:.5f} ({jointangle_loss_avg:.5f})'.format(
                                jointangle_loss = jointangle_loss.item(), jointangle_loss_avg = recorder.avg_jointangle_loss
                            )    
                        
                        logger.info(msg)
                #return recorder
                    #if i%10 == 0: break
                recorder.computeAvgLosses()
                
                global_steps = writer_dict['valid_global_steps']
                if config.LOSS.WITH_POSE3D_LOSS:
                    writer.add_scalar('val_loss/pose3d_loss', recorder.avg_pose3d_loss, global_steps)
                if config.LOSS.WITH_POSE2D_LOSS:
                    writer.add_scalar('val_loss/pose2d_loss', recorder.avg_pose2d_loss, global_steps)
                if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                    writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
                if config.LOSS.WITH_BONE_LOSS:
                    writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
                if config.LOSS.WITH_JOINTANGLE_LOSS:
                    writer.add_scalar('val_loss/jointangle_loss', recorder.avg_jointangle_loss, global_steps)

                writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)                 
                
                writer_dict['valid_global_steps'] += 1

    elif 'mhp_seq' == config.DATASET.DATASET.lower():
        if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
            with torch.no_grad():
                end = time.time()
                center_idx = config.DATASET.N_FRAMES // 2 # 27 // 2 = 13
                intrin_m = torch.from_numpy(val_loader.dataset.intrinsic_matrix.T).cuda(non_blocking=True)
                
                # reusable data
                reusable_flag = (config.DATASET.N_FRAMES > config.DATASET.SAMPLE_STRIDE)
                reusable_frames = None
                reusable_pose2d_gt = None
                reusable_pose3d_cam_gt = None
                
                for i, (frames, pose2d_gt, pose3d_cam_gt) in enumerate(val_loader):
                    # Length = min(sample_len, sample_stride)
                    # frames (have been normalized):  size = Batch x Length x 3 x H x W
                    # pose2d_gt:  size = B x Length x 21 x 3 (u,v,visibility)
                    # pose3d_gt:  size = B x Length x 21 x 3 (u,v,d) cam coord

                    first_flag = (frames.shape[1] == config.DATASET.N_FRAMES)

                    # compute output
                    #print('eval',i,':',frames.shape[1],first_flag, reusable_flag, type(reusable_frames))
                    if False:
                        entire_frames = frames if first_flag or not reusable_flag else torch.cat((reusable_frames, frames), dim=1)
                        pose3d_cam_pred_center_frame_rel = model(entire_frames) # b x 21 x 3
                        
                        # b x N_frames x 21 x 3
                        entire_pose3d_cam_gt = pose3d_cam_gt if first_flag or not reusable_flag else torch.cat((reusable_pose3d_cam_gt, pose3d_cam_gt), dim=1)
                        # b x 21 x 3
                        pose3d_cam_gt_center_frame = entire_pose3d_cam_gt[:,center_idx].cuda(non_blocking=True)
                    
                        # depth-relative to the palm. size: b x 1 x 1
                        root_depth = pose3d_cam_gt_center_frame[:,0:1,-1:]
                        # get z values of all keypoints. size b x 21 x 1
                        pose3d_cam_gt_center_frame_rel_z = pose3d_cam_gt_center_frame[:,:,-1:].clone() - root_depth
                        pose3d_cam_pred_center_frame_z = pose3d_cam_pred_center_frame_rel[:,:,-1:].clone() + root_depth
                        # b x 21 x 3
                        pose3d_cam_gt_center_frame_rel = torch.cat((pose3d_cam_gt_center_frame[:,:,0:2], pose3d_cam_gt_center_frame_rel_z), dim=2)
                        # b x 21 x 3
                        pose3d_cam_pred_center_frame = torch.cat((pose3d_cam_pred_center_frame_rel[:,:,0:2], pose3d_cam_pred_center_frame_z), dim=2)
                        
                        pose2d_reproj_center_frame_xyz = torch.matmul(
                        pose3d_cam_pred_center_frame, # b x 21 x 3
                        intrin_m # 3 x 3
                        ) # get b x 21 x 3 

                        pose2d_reproj_center_frame_uv = pose2d_reproj_center_frame_xyz[:,:,0:2] / pose2d_reproj_center_frame_xyz[:,:,2:]
                        
                        entire_pose2d_gt = pose2d_gt if first_flag or not reusable_flag else torch.cat((reusable_pose2d_gt, pose2d_gt), dim=1) 
                        pose2d_gt_center_frame_wo_vis = entire_pose2d_gt[:,center_idx,:,0:2].cuda(non_blocking=True)
                        
                        if reusable_flag:
                            start_idx = config.DATASET.SAMPLE_STRIDE
                            reusable_frames = entire_frames[:,start_idx:]
                            reusable_pose2d_gt = entire_pose2d_gt[:,start_idx:]
                            reusable_pose3d_cam_gt = entire_pose3d_cam_gt[:,start_idx:]

                        #continue
                        loss_dict = recorder.computeLosses(
                            pose2d_pred=pose2d_reproj_center_frame_uv, # b x 21 x 2
                            pose2d_gt=pose2d_gt_center_frame_wo_vis,  # b x 21 x 2
                            pose3d_pred=pose3d_cam_pred_center_frame_rel, # b x 21 x 3
                            pose3d_gt=pose3d_cam_gt_center_frame_rel,     # b x 21 x 3
                            with_visibility=False
                        )
                    else:
                        entire_frames = frames if first_flag or not reusable_flag else torch.cat((reusable_frames, frames), dim=1)
                        pose3d_cam_pred_center_frame = model(entire_frames) # b x 21 x 3
                        # b x N_frames x 21 x 3
                        entire_pose3d_cam_gt = pose3d_cam_gt if first_flag or not reusable_flag else torch.cat((reusable_pose3d_cam_gt, pose3d_cam_gt), dim=1)
                        # b x 21 x 3
                        pose3d_cam_gt_center_frame = entire_pose3d_cam_gt[:,center_idx].cuda(non_blocking=True)
                    
                        pose2d_reproj_center_frame_xyz = torch.matmul(
                            pose3d_cam_pred_center_frame, # b x 21 x 3
                            intrin_m # 3 x 3
                            ) # get b x 21 x 3 

                        pose2d_reproj_center_frame_uv = pose2d_reproj_center_frame_xyz[:,:,0:2] / pose2d_reproj_center_frame_xyz[:,:,2:]
                        
                        entire_pose2d_gt = pose2d_gt if first_flag or not reusable_flag else torch.cat((reusable_pose2d_gt, pose2d_gt), dim=1) 
                        pose2d_gt_center_frame_wo_vis = entire_pose2d_gt[:,center_idx,:,0:2].cuda(non_blocking=True)
                        
                        if reusable_flag:
                            start_idx = config.DATASET.SAMPLE_STRIDE
                            reusable_frames = entire_frames[:,start_idx:]
                            reusable_pose2d_gt = entire_pose2d_gt[:,start_idx:]
                            reusable_pose3d_cam_gt = entire_pose3d_cam_gt[:,start_idx:]

                        #continue
                        loss_dict = recorder.computeLosses(
                            pose2d_pred=pose2d_reproj_center_frame_uv, # b x 21 x 2
                            pose2d_gt=pose2d_gt_center_frame_wo_vis,  # b x 21 x 2
                            pose3d_pred=pose3d_cam_pred_center_frame, # b x 21 x 3
                            pose3d_gt=pose3d_cam_gt_center_frame,     # b x 21 x 3
                            with_visibility=False
                        )

                    total_loss=loss_dict['total_loss']
                    pose2d_loss = loss_dict['pose2d_loss']
                    pose3d_loss = loss_dict['pose3d_loss']
                    TC_loss = loss_dict['TC_loss']
                    bone_loss = loss_dict['bone_loss']
                    jointangle_loss = loss_dict['jointangle_loss']

                    if i % config.PRINT_FREQ == 0:
                        recorder.computeAvgLosses()
                        # measure elapsed time
                        batch_time = (time.time() - end) / config.PRINT_FREQ
                        end = time.time()

                        msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time:.3f}s\t' \
                        'Speed {speed:.1f} samples/s\t' \
                        'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})\t'.format(
                            i+1, len(val_loader), batch_time=batch_time,
                            speed=frames.size(0)/batch_time,
                            total_loss=total_loss.item(),
                            total_loss_avg=recorder.avg_total_loss)

                        if pose3d_loss:
                            msg += '\tPose3DLoss {pose3d_loss:.5f} ({pose3d_loss_avg:.5f})'.format(
                                pose3d_loss = pose3d_loss.item(), pose3d_loss_avg = recorder.avg_pose3d_loss
                            )

                        if pose2d_loss:
                            msg += '\tPose2DLoss {pose2d_loss:.5f} ({pose2d_loss_avg:.5f})'.format(
                                pose2d_loss = pose2d_loss.item(), pose2d_loss_avg = recorder.avg_pose2d_loss
                            )

                        if TC_loss:
                            msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                                TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
                            )
                            
                        if bone_loss:
                            msg += '\tBoneLoss {Bone_loss:.5f} ({Bone_loss_avg:.5f})'.format(
                                Bone_loss = bone_loss.item(), Bone_loss_avg = recorder.avg_bone_loss
                            )
                            
                        if jointangle_loss:
                            msg += '\tJointangle_loss {jointangle_loss:.5f} ({jointangle_loss_avg:.5f})'.format(
                                jointangle_loss = jointangle_loss.item(), jointangle_loss_avg = recorder.avg_jointangle_loss
                            )    
                        
                        logger.info(msg)
                #return recorder
                    #if i%10 == 0: break
                recorder.computeAvgLosses()
                
                global_steps = writer_dict['valid_global_steps']
                if config.LOSS.WITH_POSE3D_LOSS:
                    writer.add_scalar('val_loss/pose3d_loss', recorder.avg_pose3d_loss, global_steps)
                if config.LOSS.WITH_POSE2D_LOSS:
                    writer.add_scalar('val_loss/pose2d_loss', recorder.avg_pose2d_loss, global_steps)
                if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                    writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
                if config.LOSS.WITH_BONE_LOSS:
                    writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
                if config.LOSS.WITH_JOINTANGLE_LOSS:
                    writer.add_scalar('val_loss/jointangle_loss', recorder.avg_jointangle_loss, global_steps)

                writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)                 
                
                writer_dict['valid_global_steps'] += 1

    elif 'fha' in config.DATASET.DATASET.lower():
        with torch.no_grad():
            end = time.time()
            for i, (imgs, pose2d_gt, pose3d_gt) in enumerate(val_loader):
                # imgs (have been normalized):  size = Batch x Length x 3 x H x W
                # pose2d_gt:  size = B x Length x 21 x 3 (u,v,visibility)
                # pose3d_gt:  size = B x Length x 21 x 3 (u,v,d) cam coord

                # compute output
                pose3d_pred = model(imgs.cuda(non_blocking=True))
                pose3d_pred = scale_pose3d(pose3d_pred)

                pose3d_gt = torch.squeeze(pose3d_gt[:,-1,:,:], dim=1).cuda(non_blocking=True)
                pose3d_gt = scale_pose3d(pose3d_gt)

                # calculate losses
                loss_dict = recorder.computeLosses_for_RNN(
                    pose3d_pred,
                    pose3d_gt,
                )

                total_loss=loss_dict['total_loss']
                pose3d_loss = loss_dict['pose3d_loss']
                TC_loss = loss_dict['TC_loss']
                bone_loss = loss_dict['bone_loss']
                jointangle_loss = loss_dict['jointangle_loss']

                if i % config.PRINT_FREQ == 0:
                    recorder.computeAvgLosses_for_RNN()
                    # measure elapsed time
                    batch_time = (time.time() - end) / config.PRINT_FREQ
                    end = time.time()

                    if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
                        msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time:.3f}s\t' \
                        'Speed {speed:.1f} samples/s\t' \
                        'TotalLoss {total_loss:.5f} ({total_loss_avg:.5f})\t'.format(
                            i+1, len(val_loader), batch_time=batch_time,
                            speed=imgs.size(0)/batch_time,
                            total_loss=total_loss.item(),
                            total_loss_avg=recorder.avg_total_loss)

                        if pose3d_loss:
                            msg += '\tPose3DLoss {pose3d_loss:.5f} ({pose3d_loss_avg:.5f})'.format(
                                pose3d_loss = pose3d_loss.item(), pose3d_loss_avg = recorder.avg_pose3d_loss
                            )
                            
                        if TC_loss:
                            msg += '\tTimeConsistencyLoss {TC_loss:.5f} ({TC_loss_avg:.5f})'.format(
                                TC_loss = TC_loss.item(), TC_loss_avg = recorder.avg_time_consistency_loss
                            )
                            
                        if bone_loss:
                            msg += '\tBoneLoss {Bone_loss:.5f} ({Bone_loss_avg:.5f})'.format(
                                Bone_loss = bone_loss.item(), Bone_loss_avg = recorder.avg_bone_loss
                            )
                            
                        if jointangle_loss:
                            msg += '\tJointangle_loss {jointangle_loss:.5f} ({jointangle_loss_avg:.5f})'.format(
                                jointangle_loss = jointangle_loss.item(), jointangle_loss_avg = recorder.avg_jointangle_loss
                            )    
                        
                        logger.info(msg)

                #if i%20 == 0: break

            recorder.computeAvgLosses_for_RNN()
            
            if not config.DISTRIBUTED or config.DISTRIBUTED and args.local_rank == config.GPUS[0]:
                global_steps = writer_dict['valid_global_steps']
                if config.LOSS.WITH_POSE3D_LOSS:
                    writer.add_scalar('val_loss/pose3d_loss', recorder.avg_pose3d_loss, global_steps)
                if config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
                    writer.add_scalar('val_loss/time_consistency_loss', recorder.avg_time_consistency_loss, global_steps)
                if config.LOSS.WITH_BONE_LOSS:
                    writer.add_scalar('val_loss/bone_loss', recorder.avg_bone_loss, global_steps)
                if config.LOSS.WITH_JOINTANGLE_LOSS:
                    writer.add_scalar('val_loss/jointangle_loss', recorder.avg_jointangle_loss, global_steps)

                writer.add_scalar('val_loss/total_loss', recorder.avg_total_loss, global_steps)                 
            
            writer_dict['valid_global_steps'] += 1
    
    else:
        print('[Error in funtion.py] Invalid dataset!')
        exit()

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
        self.time_consistency_loss = 0. if config.LOSS.WITH_TIME_CONSISTENCY_LOSS else None
        self.bone_loss = 0. if config.LOSS.WITH_BONE_LOSS else None
        self.jointangle_loss = 0. if config.LOSS.WITH_JOINTANGLE_LOSS else None
        self.n = 0
    
    def computeAvgLosses(self):
        self.avg_total_loss = self.total_loss / self.n

        loss_dict = {'total_loss': self.avg_total_loss}

        if self.config.LOSS.WITH_HEATMAP_LOSS:
            self.avg_heatmap_loss = self.heatmap_loss / self.n
            loss_dict['heatmap_loss'] = self.avg_heatmap_loss

        if self.config.LOSS.WITH_POSE2D_LOSS:
            self.avg_pose2d_loss = self.pose2d_loss / self.n
            loss_dict['pose2d_loss'] = self.avg_pose2d_loss

        if self.config.LOSS.WITH_POSE3D_LOSS:
            self.avg_pose3d_loss = self.pose3d_loss / self.n
            loss_dict['pose3d_loss'] = self.avg_pose3d_loss
        
        if self.config.LOSS.WITH_TIME_CONSISTENCY_LOSS:
            self.avg_time_consistency_loss = self.time_consistency_loss / self.n
            loss_dict['TC_loss'] = self.avg_time_consistency_loss

        if self.config.LOSS.WITH_BONE_LOSS:
            self.avg_bone_loss = self.bone_loss / self.n
            loss_dict['bone_loss'] = self.avg_bone_loss

        if self.config.LOSS.WITH_JOINTANGLE_LOSS:
            self.avg_jointangle_loss = self.jointangle_loss / self.n
            loss_dict['jointangle_loss'] = self.avg_jointangle_loss

        return loss_dict


    def computeLosses(self, heatmaps_pred=None, heatmaps_gt=None, pose2d_pred=None, pose2d_gt=None, pose3d_pred=None, pose3d_gt=None, n=1, with_visibility=False):
        self.n += n
        loss_dict = {
            'heatmap_loss': None,
            'pose2d_loss': None,
            'pose3d_loss': None,
            'TC_loss': None,
            'jointangle_loss': None,
            'bone_loss': None,
            'total_loss': 0
            }

        total_loss = 0
        loss_functions = self.criterion.keys()

        if 'heatmap_loss' in loss_functions:
            heatmap_loss = self.criterion['heatmap_loss'](heatmaps_pred, heatmaps_gt)
            self.heatmap_loss += heatmap_loss.item()
            total_loss += self.config.LOSS.HEATMAP_LOSS_FACTOR * heatmap_loss
            loss_dict['heatmap_loss'] = heatmap_loss

        if 'pose2d_loss' in loss_functions:
            pose2d_loss = self.criterion['pose2d_loss'](pose2d_pred, pose2d_gt, with_visibility)
            self.pose2d_loss += pose2d_loss.item()
            total_loss += self.config.LOSS.POSE2D_LOSS_FACTOR * pose2d_loss
            loss_dict['pose2d_loss'] = pose2d_loss

        if 'pose3d_loss' in loss_functions:
            pose3d_loss = self.criterion['pose3d_loss'](pose3d_pred, pose3d_gt)
            self.pose3d_loss += pose3d_loss.item()
            total_loss += self.config.LOSS.POSE3D_LOSS_FACTOR * pose3d_loss
            loss_dict['pose3d_loss'] = pose3d_loss

        if 'bone_loss' in loss_functions or 'jointangle_loss' in loss_functions:
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
        