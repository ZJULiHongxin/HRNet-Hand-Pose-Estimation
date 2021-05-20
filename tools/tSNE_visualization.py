import argparse
import os
import time
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from torchvision import transforms
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import _init_paths
from fp16_utils.fp16util import network_to_half
from fp16_utils.fp16_optimizer import FP16_Optimizer

from config import cfg
from config import update_config
from utils.utils import get_model_summary

import dataset
from dataset import make_dataloader, make_test_dataloader
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

"""
python tSNE_visualization.py  --cfg ../experiments/RHD/w32_256x256_adam_lr1e-3.yaml
"""
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)
n_components = 3
perplexity = 30
MACHINE_EPSILON = np.finfo(np.double).eps
parser = argparse.ArgumentParser(description='Please specify the mode [training/assessment/predicting]')
parser.add_argument('--cfg',
                help='experiment configure file name',
                required=True,
                type=str)
parser.add_argument('opts',
                help="Modify config options using the command-line",
                default=None,
                nargs=argparse.REMAINDER)
parser.add_argument('--gpu',
                    help='gpu id for multiprocessing training',
                    type=str)
parser.add_argument('--world-size',
                    default=1,
                    type=int,
                    help='number of nodes for distributed training')

args = parser.parse_args()

update_config(cfg, args)

X = []

dataloader, _ = make_test_dataloader(cfg)
for i, (images, heatmaps, joints, img_path) in enumerate(dataloader):
    rel_joints = joints[0][:,0:2] - joints[0][0,0:2]
    X.append(rel_joints.numpy().ravel())

X = np.stack(X)
print(X.shape)
tsne = TSNE(n_components=n_components)
X_embedded = tsne.fit_transform(X)
print(X_embedded.shape)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2])
# sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], legend='full', palette=palette)
plt.show()