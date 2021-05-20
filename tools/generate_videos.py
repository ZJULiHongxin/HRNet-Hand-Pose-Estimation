# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import time


import cv2
import numpy as np

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, LossFactory
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import create_logger
from utils.utils import setup_logger
from utils.utils import get_model_summary
from fp16_utils.fp16util import network_to_half
from fp16_utils.fp16_optimizer import FP16_Optimizer

import dataset
from dataset import Frei

def main():
    parser = argparse.ArgumentParser(description='Please specify the mode [training/assessment/predicting]')
    parser.add_argument('--root',
                    help='experiment configure file name',
                    required=True,
                    type=str)
    parser.add_argument('--mode',
                    help='experiment configure file name',
                    required=True,
                    type=str)
    args = parser.parse_args()

    root = args.root
    mode = args.mode
    _dataset = Frei(root, mode, cfg.DATASET.DATA_FORMAT)
    print('Loading dataset:', root+'\\'+mode)
    _dataset.generate_videos()
    # create a metrics recorder

if __name__ == '__main__':
    # python generate_videos.py --root E:\Hand_Datasets\FreiHAND_pub_v2 --mode training
    main()