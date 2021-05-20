# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class HeatmapLoss(nn.Module):
    def __init__(self, mode='l2'):
        super().__init__()
        self.mode = mode
    def forward(self, pred, gt):
        # pred: b (x n_joints) x H x W
        assert pred.size() == gt.size(), \
            'Heatmap loss error: prediced heatmaps have size {}, but the groundtruth has {}'.format(pred.shape, gt.shape)
        if self.mode == 'l2':
            loss = ((pred - gt)**2).expand_as(pred) # tensor.expand_as()这个函数就是把一个tensor变成和函数括号内一样形状的tensor
        elif self.mode == 'l1':
            loss = torch.abs(pred - gt)
        mean_loss = loss.sum(dim=-1).sum(dim=-1).mean()
        return mean_loss

class JointsMSELoss(nn.Module):
    """
    Mean Square Error
    """
    def __init__(self):
        super(JointsMSELoss, self).__init__()

    def forward(self, pose2D_pred, pose2D_gt, visibility=None):
        """
        params:
        - pose2D_pred:  B x 21 x 2
        - pose2D_gt:    B x 21 x 2
        - visibility:   B x 21
        """

        if visibility is not None:
            pose2d_loss = torch.norm(pose2D_pred - pose2D_gt, dim=2) * visibility
            pose2d_loss_mean = torch.sum(pose2d_loss) / max(1, torch.sum(visibility))
            return pose2d_loss_mean
        else:
            return torch.norm(pose2D_pred - pose2D_gt, dim=2).sum() / pose2D_pred.shape[1] # a scalar

class JointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, pose_pred, pose_gt, visibility=None):
        # pose_pred/gt: batch_size x N_joints x 2/3
        # visibility: batch_size x N_joints
        if visibility:
            diff = (pose_gt - pose_pred) ** 2 * visibility
            diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
            loss = torch.sum(diff) / max(1, torch.sum(visibility).item())
        else:
            diff = (pose_gt - pose_pred) ** 2
            diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
            loss = torch.sum(diff) / pose_gt.shape[1]
        return loss

class JointsMAELoss(nn.Module):
    """
    Mean Absolute Error
    """
    def __init__(self):
        super().__init__()

    def forward(self, pose_pred, pose_gt, visibility=None):
        """
        params:
        - pose2D_pred:  B x 21 x 2/3
        - pose2D_gt:    B x 21 x 2/3 
        - visibility:   B x 21 x 1
        """
        if visibility is not None:
            loss = torch.sum(torch.abs(pose_gt - pose_pred) * visibility)
            loss = loss / max(1, torch.sum(visibility).item())
        else:
            loss = torch.sum(torch.abs(pose_gt - pose_pred)) / pose_gt.shape[1]

        return loss

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

class Joints3DMSELoss(nn.Module):
    def __init__(self):
        super(Joints3DMSELoss, self).__init__()

    def forward(self, pose3d_pred, pose3d_gt):
        """
        params:
        - pose3d_pred:  B x 21 x 3 [u,v,d] (cam coord)
        - pose3d_gt:    B x 21 x 3 [u,v,d] (cam coord)
        """
 
        return torch.norm(pose3d_gt - pose3d_pred, dim=2).sum() / pose3d_pred.shape[1] # a scalar

class BoneLengthLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pose23d_pred, pose23d_gt):
        """
        params:
        - pose23d_pred:  B x 21 x 3/2 [u,v,d]/[u,v] (cam coord)/(image coord)
        - pose23d_gt:    B x 21 x 3/2 [u,v,d]/[u,v] (cam coord)/(image coord)
        """

        loss = 0.

        # 0-1,2,3,4; 0-5,6,7,8; ...
        for b in range(pose23d_pred.shape[0]):
            for finger_idx in range(0,5):
                for joint_idx in range(finger_idx*4+1, finger_idx*4+5):
                    if joint_idx == finger_idx:
                        bone_len_gt = torch.norm(pose23d_gt[b][joint_idx] - pose23d_gt[b][0])
                        bone_len_pred = torch.norm(pose23d_pred[b][joint_idx] - pose23d_pred[b][0])
                    else:
                        bone_len_gt = torch.norm(pose23d_gt[b][joint_idx] - pose23d_gt[b][joint_idx-1])
                        bone_len_pred = torch.norm(pose23d_pred[b][joint_idx] - pose23d_pred[b][joint_idx-1])
                            
                    loss += (bone_len_gt - bone_len_pred) ** 2


        return loss / 20

class JointAngleLoss(nn.Module):
    """
    Reference: 2019 End-to-End
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pose23d_pred):
        """
        params:
        - pose23d_pred:  B x 21 x 3 [u,v,d] (cam coord) or B x 21 x 2 [u,v] (img coord)
        """
        if pose23d_pred.shape[2] == 3:
            use_coplane_reg = True

        loss = 0.
        for b in range(pose23d_pred.shape[0]):
            for finger_idx in range(0,5):
                # note: torch.cross only applys to matrices, hence torch.unsqueeze
                bone1 = torch.unsqueeze(pose23d_pred[b][finger_idx*4+1] - pose23d_pred[b][finger_idx*4], 0)
                bone2 = torch.unsqueeze(pose23d_pred[b][finger_idx*4+2] - pose23d_pred[b][finger_idx*4+1], 0)
                bone3 = torch.unsqueeze(pose23d_pred[b][finger_idx*4+3] - pose23d_pred[b][finger_idx*4+2], 0)
                bone4 = torch.unsqueeze(pose23d_pred[b][finger_idx*4+4] - pose23d_pred[b][finger_idx*4+3], 0)
                
                # Three rotation vectors representing the rotating directions of finger joints
                rot_vec_near_tip = torch.squeeze(torch.cross(bone4, bone3))
                rot_vec_near_middle = torch.squeeze(torch.cross(bone3, bone2))
                rot_vec_near_paml = torch.squeeze(torch.cross(bone2, bone1))

                # Rule 1: The four joints of each finger should in the same plane
                if use_coplane_reg:
                    loss = loss + torch.dot(rot_vec_near_paml, torch.squeeze(bone4)) \
                        + torch.dot(rot_vec_near_middle, torch.squeeze(bone4))
                
                # Rule 2: The rotating directions should be consistent
                vec_dot_1 = torch.dot(rot_vec_near_tip, rot_vec_near_middle)
                vec_dot_2 = torch.dot(rot_vec_near_paml, rot_vec_near_middle)
                
                # || min(dot_product, 0) ||^2
                if vec_dot_1 < 0:
                    loss += vec_dot_1 ** 2
                if vec_dot_2 < 0:
                    loss += vec_dot_2 ** 2

        return loss

class VolumetricCELoss(nn.Module):
    """
    A regularization item
    """
    def __init__(self):
        super().__init__()

    def forward(self, coord_volumes_batch, volumes_batch_pred, keypoints_gt, keypoints_binary_validity):
        loss = 0.0
        n_losses = 0

        batch_size = volumes_batch_pred.shape[0]
        for batch_i in range(batch_size):
            coord_volume = coord_volumes_batch[batch_i]
            keypoints_gt_i = keypoints_gt[batch_i]

            coord_volume_unsq = coord_volume.unsqueeze(0)
            keypoints_gt_i_unsq = keypoints_gt_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
            dists = dists.view(dists.shape[0], -1)

            min_indexes = torch.argmin(dists, dim=-1).detach().cpu().numpy()
            min_indexes = np.stack(np.unravel_index(min_indexes, volumes_batch_pred.shape[-3:]), axis=1)

            for joint_i, index in enumerate(min_indexes):
                validity = keypoints_binary_validity[batch_i, joint_i]
                loss += validity[0] * (-torch.log(volumes_batch_pred[batch_i, joint_i, index[0], index[1], index[2]] + 1e-6))
                n_losses += 1


        return loss / n_losses

