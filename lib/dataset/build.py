# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .RHDDataset import RHDDataset as RHD
from .RHDDatasetKeypoints import RHDDataset_Keypoint as RHD_kpt
from .FreiHandDataset import FreiHandDataset as FreiHand
from .FreiHandDatasetKeypoints import FreiHandDataset_Keypoint as FreiHand_kpt
from .HandGraphDataset import HandGraphDataset as HandGraph
from .HandGraphDatasetKeypoints import HandGraphDataset_Keypoint as HandGraph_kpt
from .FHADataset import FHADataset as FHA
from .FHADatasetKeypoints import FHADataset_Keypoint as FHA_kpt
from .MHPDataset import MHPDataset as MHP
from .MHPDatasetKeypoints import MHPDataset_keypoint as MHP_kpt
from .MHPSeqDataset import MHPSeqDataset as MHP_seq
from .MHPMultiViewDataset import MHPMultiViewDataset as MHP_mv
from .MHP_CPMDataset import MHP_CPMDataset as MHP_CPM_kpt
from .MHP_CPMMultiViewDataset import MHP_CPMMultiViewDataset as MHP_CPM_mv
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import ScaleAwareHeatmapGenerator

def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    if cfg.DATASET.SCALE_AWARE_SIGMA:
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS * cfg.DATASET.N_FRAMES, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE #[64]
    ]

    if is_train:
        dataset_list = cfg.DATASET.DATASET
        subset = cfg.DATASET.TRAIN_SET # 'training'
    else:
        dataset_list = cfg.DATASET.TEST_DATASET
        subset = cfg.DATASET.TEST_SET  # 'evaluation'

    # pack all datasets in a dict for the convenience of joint dataset training
    dataset_dict = {}
    for dataset in dataset_list:
        dataset_dict[dataset] = eval(dataset)(
                                            config=cfg,
                                            set_name=subset,
                                            heatmap_generator=heatmap_generator[0],
                                            transforms=transforms
                                            )

    return dataset_dict


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    
    dataset_dict = build_dataset(cfg, is_train)

    data_loader_dict = {}
    for name, dataset in dataset_dict.items():
        if is_train and distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset
            )
            shuffle = False
            images_per_batch = images_per_gpu
        else:
            images_per_batch = images_per_gpu * len(cfg.GPUS)
            train_sampler = None
        
        data_loader_dict[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=images_per_batch,
            shuffle=shuffle,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            sampler=train_sampler
        )

    return data_loader_dict


def make_test_dataloader(cfg):
    transforms = build_transforms(cfg, is_train=False)

    if cfg.DATASET.SCALE_AWARE_SIGMA:
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.TEST_SET,
        heatmap_generator[0],
        transforms
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
# def build_dataset(cfg, is_train):
#     transforms = build_transforms(cfg, is_train)

#     if cfg.DATASET.SCALE_AWARE_SIGMA:
#         _HeatmapGenerator = ScaleAwareHeatmapGenerator
#     else:
#         _HeatmapGenerator = HeatmapGenerator

#     heatmap_generator = [
#         _HeatmapGenerator(
#             output_size, cfg.DATASET.NUM_JOINTS * cfg.DATASET.N_FRAMES, cfg.DATASET.SIGMA
#         ) for output_size in cfg.DATASET.OUTPUT_SIZE #[64]
#     ]

#     dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET

#     dataset = eval(cfg.DATASET.DATASET)(
#         cfg,
#         dataset_name,
#         heatmap_generator[0],
#         transforms
#     )

#     return dataset


# def make_dataloader(cfg, is_train=True, distributed=False):
#     if is_train:
#         images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
#         shuffle = True
#     else:
#         images_per_gpu = cfg.TEST.IMAGES_PER_GPU
#         shuffle = False
    

#     dataset = build_dataset(cfg, is_train)

#     if is_train and distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(
#             dataset
#         )
#         shuffle = False
#         images_per_batch = images_per_gpu
#     else:
#         images_per_batch = images_per_gpu * len(cfg.GPUS)
#         train_sampler = None

#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=images_per_batch,
#         shuffle=shuffle,
#         num_workers=cfg.WORKERS,
#         pin_memory=cfg.PIN_MEMORY,
#         sampler=train_sampler
#     )

#     return data_loader


# def make_test_dataloader(cfg):
#     transforms = build_transforms(cfg, is_train=False)

#     if cfg.DATASET.SCALE_AWARE_SIGMA:
#         _HeatmapGenerator = ScaleAwareHeatmapGenerator
#     else:
#         _HeatmapGenerator = HeatmapGenerator

#     heatmap_generator = [
#         _HeatmapGenerator(
#             output_size, cfg.DATASET.NUM_JOINTS, cfg.DATASET.SIGMA
#         ) for output_size in cfg.DATASET.OUTPUT_SIZE
#     ]

#     dataset = eval(cfg.DATASET.DATASET)(
#         cfg,
#         cfg.DATASET.TEST_SET,
#         heatmap_generator[0],
#         transforms
#     )

#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=False
#     )

#     return data_loader, dataset
