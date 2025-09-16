import math
import warnings
from functools import partial

import timm
import torch
import torch.nn as nn

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_


logger = logging.getLogger("radio")

class RADIOFeaturizer(nn.Module):

    def __init__(self, arch="radio_v2.5-b", patch_size=16, feat_type=None):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type

        self.n_feats = 128
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version="radio_v2.5-b", progress=True, skip_validation=True)
        self.model.make_preprocessor_external()

    def forward(self, img, n=1, include_cls=False):
        summary, spatial_features = self.model(img, feature_fmt='NCHW')
        return spatial_features
