# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision
from .dota import build as build_dota

def build_dataset(image_set, args):
    if args.dataset_file == "dota1.5" or args.dataset_file == "dota1.0":
        return build_dota(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
