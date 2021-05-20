# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size(), \
            'Heatmap loss error: prediced heatmaps have size {}, but the groundtruth has {}'.format(pred.shape, gt.shape)
        loss = ((pred - gt)**2).expand_as(pred) # tensor.expand_as()这个函数就是把一个tensor变成和函数括号内一样形状的tensor
        mean_loss = loss.sum(dim=3).sum(dim=2).mean()
        return mean_loss

class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()

    def forward(self, pose2D_pred, pose2D_gt, with_visibility=False):
        """
        params:
        - pose2D_pred:  B x 21 x 2
        - pose2D_gt:    B x 21 x 2 or B x 21 x 3 (with visibility)
        """
        if with_visibility:
            pose2d_loss = torch.norm(pose2D_pred - pose2D_gt[:,:,0:2], dim=2) * pose2D_gt[:,:,2]
            pose2d_loss_mean = torch.sum(pose2d_loss) / torch.sum(pose2D_gt[:,:,2])  
            return pose2d_loss_mean
        else:
            return torch.norm(pose2D_pred - pose2D_gt, dim=2).sum() / pose2D_pred.shape[1] # a scalar

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
    
    def forward(self, pose23d_pred, pose23d_gt):
        """
        params:
        - pose23d_pred:  B x 21 x 3 [u,v,d] (cam coord) or B x 21 x 2 [u,v] (img coord)
        - pose23d_gt:    B x 21 x 3 [u,v,d] (cam coord) or B x 21 x 2 [u,v] (img coord)
        """

        loss = 0.
        for b in range(pose23d_pred.shape[0]):
            for finger_idx in range(0,5):
                # note: torch.cross only applys to matrices, hence torch.unsqueeze
                bone1 = torch.unsqueeze(pose23d_gt[b][finger_idx*4+1] - pose23d_gt[b][finger_idx*4], 0)
                bone2 = torch.unsqueeze(pose23d_gt[b][finger_idx*4+2] - pose23d_gt[b][finger_idx*4+1], 0)
                bone3 = torch.unsqueeze(pose23d_gt[b][finger_idx*4+3] - pose23d_gt[b][finger_idx*4+2], 0)
                bone4 = torch.unsqueeze(pose23d_gt[b][finger_idx*4+4] - pose23d_gt[b][finger_idx*4+3], 0)
                
                # Three rotation vectors representing the rotating directions of finger joints
                rot_vec_near_tip = torch.squeeze(torch.cross(bone4, bone3))
                rot_vec_near_middle = torch.squeeze(torch.cross(bone3, bone2))
                rot_vec_near_paml = torch.squeeze(torch.cross(bone2, bone1))

                # Rule 1: The four joints of each finger should in the same plane
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

class LossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.heatmaps_loss = None
        self.pose2D_loss = None
        self.bone_loss = None

        self.heatmaps_loss_factor = 1.0

        if cfg.LOSS.WITH_HEATMAP_LOSS:
            self.heatmaps_loss = HeatmapLoss()
            self.heatmaps_loss_factor = cfg.LOSS.HEATMAP_LOSS_FACTOR

        if cfg.LOSS.WITH_POSE2D_LOSS:
            self.pose2D_loss = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
            self.pose2D_loss_factor = cfg.LOSS.POSE2D_LOSS_FACTOR
        
        if not self.heatmaps_loss and not self.pose2D_loss:
            logger.error('At least enable one loss!')

    def forward(self, pred_heatmaps, target_heatmaps, target_joints):
        """
        params:
        - outputs: predicted heat maps of size B x 21 x H x W
        - heatmaps
        """
        if self.heatmaps_loss is not None:
            heatmaps_loss = self.heatmaps_loss(pred_heatmaps, target_heatmaps)
            heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor

        # if self.pose2D_loss is not None:
        #     pose2D_loss = self.pose2D_loss()

        return heatmaps_loss


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # init check
        self._init_check(cfg)

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_stages = cfg.LOSS.NUM_STAGES

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in cfg.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss(cfg.LOSS.AE_LOSS_TYPE) if with_ae_loss else None
                    for with_ae_loss in cfg.LOSS.WITH_AE_LOSS
                ]
            )
        self.push_loss_factor = cfg.LOSS.PUSH_LOSS_FACTOR
        self.pull_loss_factor = cfg.LOSS.PULL_LOSS_FACTOR

    def forward(self, outputs, heatmaps, joints):
        # forward check
        self._forward_check(outputs, heatmaps, joints)

        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints

                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred, heatmaps[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](
                    tags_pred, joints[idx]
                )
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)

        return heatmaps_losses, push_losses, pull_losses

    def _init_check(self, cfg):
        assert isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)), \
            'LOSS.WITH_HEATMAPS_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)), \
            'LOSS.HEATMAPS_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.WITH_AE_LOSS, (list, tuple)), \
            'LOSS.WITH_AE_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == len(cfg.LOSS.HEATMAPS_LOSS_FACTOR), \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.HEATMAPS_LOSS_FACTOR should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), len(cfg.LOSS.HEATMAPS_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_AE_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_AE_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PUSH_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PUSH_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PUSH_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PULL_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PULL_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PULL_LOSS_FACTOR))

    def _forward_check(self, outputs, heatmaps, joints):
        assert isinstance(outputs, list), \
            'outputs should be a list, got {} instead.'.format(type(outputs))
        assert isinstance(heatmaps, list), \
            'heatmaps should be a list, got {} instead.'.format(type(heatmaps))
        # assert isinstance(masks, list), \
        #     'masks should be a list, got {} instead.'.format(type(masks))
        assert isinstance(joints, list), \
            'joints should be a list, got {} instead.'.format(type(joints))
        assert len(outputs) == self.num_stages, \
            'len(outputs) and num_stages should been same, got {} vs {}.'.format(len(outputs), self.num_stages)
        assert len(outputs) == len(heatmaps), \
            'outputs and heatmaps should have same length, got {} vs {}.'.format(len(outputs), len(heatmaps))
        # assert len(outputs) == len(masks), \
        #     'outputs and masks should have same length, got {} vs {}.'.format(len(outputs), len(masks))
        assert len(outputs) == len(joints), \
            'outputs and joints should have same length, got {} vs {}.'.format(len(outputs), len(joints))
        assert len(outputs) == len(self.heatmaps_loss), \
            'outputs and heatmaps_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.heatmaps_loss))
        assert len(outputs) == len(self.ae_loss), \
            'outputs and ae_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.ae_loss))