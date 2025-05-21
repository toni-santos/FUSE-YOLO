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
import torch.nn.functional as F

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

class CFT(nn.Module):
    """
    Cross-Modality Fusion Transformer (CFT) module that fuses N features using a transformer architecture.
    Based on the work in https://arxiv.org/pdf/2111.00273.
    """

    # NOTE: The original CFT needs a decoder, but for this implementation it's not needed.
    def __init__(self, c1, c2, n_heads=8, n_layers=8):
        super(CFT, self).__init__()

        # Transformer layers
        self.transformer_block = TransformerBlock(c1, c2, n_heads, n_layers)

    def forward(self, x):
        """
        Args:
            x (list): List of feature maps to be fused. Each feature map should have shape (batch_size, channels, height, width).
        """

        bs, c, h, w = x[0].shape
        
        # x = torch.cat([f.view(bs, c, -1) for f in x])  # flatten and concat the features
        # x = x.permute(0, 2, 1).contiguous()
        # x = self.dropout(self.position_embedding + x)
        x = torch.cat(x, dim=1)
        x = self.transformer_block(x)  # apply transformer block

        return x

# Attention Modules
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=9):
        super().__init__()
        print("ChannelAttention", in_channels, reduction, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))

        return torch.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return torch.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=9):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, 3)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class MCBAM(nn.Module):
    """
    Multi-input CBAM module that applies the CBAM attention mechanism to a list of feature maps.
    Concatenates the feature maps along the channel dimension and applies the CBAM module.
    """

    def __init__(self, in_channels, reduction=9):
        super().__init__()
        self.cbam = CBAM(in_channels, reduction)

    def forward(self, x):
        x = torch.cat(x, dim=1)  # Concatenate the feature maps along the channel dimension
        return self.cbam(x)

class CBAMC(nn.Module):
    """
    Multi-input CBAM module that applies the CBAM attention mechanism to a list of feature maps.
    Applies the CBAM module to each feature map individually and then concatenates the results.
    """

    def __init__(self, in_channels, out_channels, ni, reduction=3):
        super().__init__()
        self.cbam = CBAM(in_channels, reduction)
        self.conv = nn.Conv2d(in_channels * ni, out_channels, kernel_size=1)

    def forward(self, x):
        r = []
        for i in range(len(x)):
            r.append(self.cbam(x[i]))
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        return x

class MCBAMC(nn.Module):
    """
    - Deprecated
    Multi-input CBAM module that applies the CBAM attention mechanism to a list of feature maps.
    Applies the CBAM module to each feature map individually and then concatenates the results.
    """

    def __init__(self, in_channels, out_channels, ni, reduction=3):
        super().__init__()
        self.cat = Concat(dimension=1)
        self.conv = nn.Conv2d(in_channels * ni, out_channels, kernel_size=1)
        self.cbam = CBAM(in_channels, reduction)

    def forward(self, x):
        return self.cbam(self.conv(self.cat(x, dim=1)))


class Trans(nn.Module):
    def __init__(self, in_channels, out_channels, img_size = 512, num_heads=3, num_blocks=8):
        super().__init__()
        self.in_channels = in_channels * 3
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        
        # Ensure num_heads divides embed_dim
        if self.in_channels % num_heads != 0:
            num_heads = 3  # Fallback to 3 if not divisible
        
        self.embed = PatchEmbed(in_chans=self.in_channels, embed_dim=self.in_channels, img_size=(img_size, img_size), patch_size=(16, 16), multi_conv=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * img_size, self.in_channels))
        self.linears = nn.ModuleList([
            nn.Linear(self.in_channels, self.in_channels) for _ in range(3)
        ])
        self.mha = nn.MultiheadAttention(embed_dim=self.in_channels, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(self.in_channels)
        self.norm2 = nn.LayerNorm(self.in_channels)
        self.dropout0 = nn.Dropout(0.1)

        # Make sure FFN outputs in_channels, not out_channels for residual connection
        self.ffn = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels * 2),  # Reduce expansion factor
            nn.ReLU(),
            nn.Linear(self.in_channels * 2, self.in_channels)  # Output same as input dims
        )
        
        # Final projection to target output channels
        self.proj = nn.Linear(self.in_channels, self.out_channels)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, x):
        """
        x -> [AIA_211, AIA_335, HMI_Ic]
        """
        b, c, h, w = x[0].shape

        AIA_211 = x[0]
        AIA_335 = x[1]
        HMI_Ic = x[2]

        # Concatenate along the channel dimension
        x = torch.cat([AIA_211, AIA_335, HMI_Ic], dim=1)        
        x = self.dropout0(self.embed(x) + self.pos_embed)
        
        x = self.norm1(x)

        for _ in range(self.num_blocks):
            [q, k, v] = [linear(x) for linear in self.linears]

            attn_out = self.mha(q, k, v)[0]
            attn_out = self.dropout1(attn_out)
            x_res1 = x + attn_out
            x_norm2 = self.norm2(x_res1)

            ffn_out = self.ffn(x_norm2)
            ffn_out = self.dropout2(ffn_out)
            x = x_res1 + ffn_out

        x_proj = self.proj(x)
        patch_size = 16  # Based on your PatchEmbed config
        out_h = h // patch_size
        out_w = w // patch_size
        
        # Reshape properly: [b, seq_len, out_c] -> [b, out_c, out_h, out_w]
        x_out = x_proj.transpose(1, 2).reshape(b, self.out_channels, out_h, out_w)

        # Upsample
        x_out = F.interpolate(x_out, size=(h, w), mode='bilinear', align_corners=False)
        
        return x_out

# NOTE: WIP
class Trans2(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, img_size=512):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Ensure num_heads divides embed_dim
        if self.in_channels % num_heads != 0:
            num_heads = 3  # Fallback to 3 if not divisible
        
        self.embeds = nn.ModuleList([
            PatchEmbed(in_chans=self.in_channels, embed_dim=self.in_channels, img_size=(img_size, img_size), patch_size=(16, 16), multi_conv=True) for _ in range(3)
        ])
        self.pos_embeds = [
            nn.Parameter(torch.zeros(1, 2 * img_size * img_size, self.in_channels)) for _ in range(3)
        ]
        self.mhas = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.in_channels, num_heads=num_heads, batch_first=True) for _ in range(2)
        ])
        self.norms = [
            nn.LayerNorm(self.in_channels),
            nn.LayerNorm(self.in_channels),
            [nn.LayerNorm(self.in_channels) for _ in range(2)]
        ]

        # Make sure FFN outputs in_channels, not out_channels for residual connection
        self.ffn = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels * 2),  # Reduce expansion factor
            nn.ReLU(),
            nn.Linear(self.in_channels * 2, self.in_channels)  # Output same as input dims
        )
        
        # Final projection to target output channels
        self.proj = nn.Linear(self.in_channels, self.out_channels)
        
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        """
        x -> [AIA_211, AIA_335, HMI_Ic]
        """
        b, c, h, w = x[0].shape

        # embeddings
        tmp = []
        for i in range(3):
            t = self.embeds[i](x[i])
            tmp.append(t + self.pos_embeds[i])
        
        # normalization
        nor = []
        for i in range(3):
            norm = self.norms[i]
            nor.append([norm(tmp[i]) for norm in norm])

        AIA_211 = nor[0]
        AIA_335 = nor[1]
        HMI = nor[2]

        attn_HMI_AIA211 = self.mhas[0](AIA_211, HMI, HMI)[0]
        attn_HMI_AIA335 = self.mhas[0](AIA_335, HMI, HMI)[0]

        attn_out = attn_HMI_AIA211 @ attn_HMI_AIA335
        attn_out = self.dropout(attn_out)
        ffn_out = self.ffn(attn_out)

        print("ffn_out", ffn_out.shape)

        patch_size = 16  # Based on your PatchEmbed config
        out_h = h // patch_size
        out_w = w // patch_size
        
        # Reshape properly: [b, seq_len, out_c] -> [b, out_c, out_h, out_w]
        # x_out = x_proj.transpose(1, 2).reshape(b, self.out_channels, out_h, out_w)

        # Upsample
        # x_out = F.interpolate(x_out, size=(h, w), mode='bilinear', align_corners=False)
        
        return ffn_out


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    from: https://github.com/IBM/CrossViT/blob/main/models/crossvit.py#L36
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
