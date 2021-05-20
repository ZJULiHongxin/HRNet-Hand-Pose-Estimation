import numpy as np
import torch

std_legend_lst = np.array([
    # 0           1               2                  3                4
    'wrist', 'thumb palm', 'thumb near palm', 'thumb near tip', 'thumb tip',
    # 5                    6                 7                8
    'index palm', 'index near palm', 'index near tip', 'index tip',
    # 9                    10                  11               12
    'middle palm', 'middle near palm', 'middle near tip', 'middle tip',
    # 13                  14               15            16
    'ring palm', 'ring near palm', 'ring near tip', 'ring tip',
    # 17                  18               19              20
    'pinky palm', 'pinky near palm', 'pinky near tip', 'pinky tip'])

# reorder indices
idx_RHD = [0,4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17]

idx_Frei = [i for i in range(21)]
idx_HandGraph = idx_Frei
idx_FHA = idx_Frei

# MPH has no wrist annotation, but palm normal instead
# hear regard palm normal as wrist
# idx_MPH = [8,5,7,6,
#             12,9,11,10,
#             20,17,19,18,
#             16,13,15,14,
#             4,1,3,2,0]

idx_MHP = [20,17,16,18,19,
            1,0,2,3,
            5,4,6,7,
            13,12,14,15,
            9,8,10,11]

# kinematic chain
KC_matrix = torch.zeros((20,21))
KC_matrix[[0,4,8,12,16],0] = -1
for k in range(20):
    KC_matrix[k,k+1] = 1
    if k % 4 != 0:
        KC_matrix[k,k] = -1