import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from models.common import *

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os

    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode

# TODO: test and propperly implement
# NOTE: after testing and implementing, turns out it doesn't work, check the DISS repository changelog for more info
class MLF(nn.Module):
    """
    Multilevel feature fusion module that integrates RGB and IR features across multiple spatial scales using 1x1 convs and depthwise separable convs.
    Based on the work in https://www.mdpi.com/2079-9292/13/2/443.
    """
    def __init__(self, c1, c2, mid_channels=64):
        super(MLF, self).__init__()
        
        # Initial 1x1 convolutions for RGB and IR inputs
        self.rgb_conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.ir_conv = nn.Conv2d(c1, mid_channels, kernel_size=1)
        
        # 1x1 convolution after concatenation
        self.concat_conv = nn.Conv2d(2*mid_channels, mid_channels, kernel_size=1)
        
        # The two 3x3 depth-wise separable convolutions
        self.upper_dwconv = DWConv(mid_channels, mid_channels, k=3)
        self.middle_dwconv = DWConv(mid_channels, mid_channels, k=3)
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(3*mid_channels, c2, kernel_size=1)
        
    def forward(self, features):
        rgb_features, ir_features = features

        # Initial processing with 1x1 convolutions
        rgb_feat = self.rgb_conv(rgb_features)
        ir_feat = self.ir_conv(ir_features)
        
        # Concatenate RGB and IR features
        concat_feat = torch.cat([rgb_feat, ir_feat], dim=1)
        
        # Process concatenated features with 1x1 convolution
        processed_feat = self.concat_conv(concat_feat)
        
        # Split into three branches
        # top, mid, bot = torch.split(processed_feat, 2, dim=1)
        top = mid = bot = processed_feat

        # Middle branch: DWConv
        middle = self.middle_dwconv(mid)
        mid_top = middle + top

        # Upper branch: DWConv
        upper = self.upper_dwconv(mid_top)        
        
        # Lower branch: Direct pass
        lower = bot

        # Concatenate the three branches
        final_added = torch.cat([upper, middle, lower], dim=1)
        
        # Final 1x1 convolution
        output = self.final_conv(final_added)
        
        return output

class CatFuse(nn.Module):
    """
    Simple concatenation fusion module that concatenates two feature maps along the channel dimension and applies a 1x1 convolution (c1 to c2).
    """

    def __init__(self, c1, c2):
        super(CatFuse, self).__init__()
        self.cat = Concat(dimension=1)
        self.conv = Conv(c1*3, c2, 1)

    def forward(self, x):
        return self.conv(self.cat(x))
