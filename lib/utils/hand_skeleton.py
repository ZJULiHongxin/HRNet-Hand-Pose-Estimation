# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import numpy as np


class Hand(object):

    def __init__(self):
        self.skeleton = self.get_skeleton()
        self.skeleton_sorted_by_level = self.sort_skeleton_by_level(
            self.skeleton)

    def get_skeleton(self):
        joint_names = [
            # 0           1               2                  3                4
            'wrist', 'thumb palm', 'thumb near palm', 'thumb near tip', 'thumb tip',
            # 5                    6                 7                8
            'index palm', 'index near palm', 'index near tip', 'index tip',
            # 9                    10                  11               12
            'middle palm', 'middle near palm', 'middle near tip', 'middle tip',
            # 13                  14               15            16
            'ring palm', 'ring near palm', 'ring near tip', 'ring tip',
            # 17                  18               19              20
            'pinky palm', 'pinky near palm', 'pinky near tip', 'pinky tip']
        
        children = [
            [1, 5, 9, 13, 17],
            [2], [3], [4], [],
            [6], [7], [8], [],
            [10], [11], [12], [],
            [14], [15], [16], [],
            [18], [19], [20], []]

        skeleton = []
        for i in range(len(joint_names)):
            skeleton.append({
                'idx': i,
                'name': joint_names[i],
                'children': children[i]
            })
        return skeleton

    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)

        queue = [skeleton[0]]
        while queue:
            cur = queue[0]
            for child in cur['children']:
                skeleton[child]['parent'] = cur['idx']
                level[child] = level[cur['idx']] + 1
                queue.append(skeleton[child])
            del queue[0]

        desc_order = np.argsort(level)[::-1]
        sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i]
            sorted_skeleton.append(skeleton[i])
        return sorted_skeleton

    def compute_limb_length(self, pose):
        """
        - pose: n_joints x 3
        """
        limb_length = {}
        skeleton = self.skeleton
        for node in skeleton:
            idx = node['idx']
            children = node['children']

            for child in children:
                length = np.linalg.norm(pose[idx] - pose[child])
                limb_length[(idx, child)] = length
        return limb_length

if __name__ == '__main__':
    hand = Hand()
    print(hand.skeleton)
    print(hand.skeleton_sorted_by_level)
