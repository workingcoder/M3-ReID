# ------------------------------------------------------------------------------
# File:    M3-ReID/models/modules/mvl_attention.py
#
# Description:
#    This module implements the Multi-View Learning (MVL) Attention mechanism.
#    It captures diverse spatio-temporal patterns by applying attention across
#    three different views: spatial (H, W), temporal-width (T, W), and
#    temporal-height (T, H).
#
# Key Features:
# - Generates multi-head attention masks for each of the three views.
# - Aggregates features using Generalized Mean (GeM) pooling or Average pooling.
# - Returns both the concatenated feature embeddings and the raw attention masks
#   (used for the Diverse Attention Constraint loss).
#
# Classes:
# - MultiViewLearningAttention
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewLearningAttention(nn.Module):
    """
    Implements the Multi-View Learning (MVL) module of M3-ReID.
    This module enhances feature representation by adaptively focusing on key
    regions and motion cues across spatial and temporal dimensions.

    'M3-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-Infrared Person Re-Identification'
    by Liang et al. See https://ieeexplore.ieee.org/document/11275868 (IEEE TIFS).
    """

    def __init__(self, in_channels, num_heads=2, mode='gem'):
        """
        Initialize the MVL Attention module.

        Args:
            in_channels (int): Number of input feature channels.
            num_heads (int): Number of attention heads per view (default 2).
            mode (str): Pooling mode for feature aggregation ('gem' or 'avg').
        """

        super(MultiViewLearningAttention, self).__init__()
        self.num_heads = num_heads
        self.mode = mode

        # Separate 1x1 Convs to generate attention masks for different dimensions
        self.view_attention_hw = nn.Conv2d(in_channels, num_heads, kernel_size=1, stride=1, padding=0, bias=True)
        self.view_attention_tw = nn.Conv2d(in_channels, num_heads, kernel_size=1, stride=1, padding=0, bias=True)
        self.view_attention_th = nn.Conv2d(in_channels, num_heads, kernel_size=1, stride=1, padding=0, bias=True)

        torch.nn.init.constant_(self.view_attention_hw.bias, 0.0)
        torch.nn.init.constant_(self.view_attention_tw.bias, 0.0)
        torch.nn.init.constant_(self.view_attention_th.bias, 0.0)

        self.activation = nn.Sigmoid()

    def gempool(self, x, p=3.0):
        """
        Performs Generalized Mean (GeM) Pooling.

        Args:
            x (Tensor): Input tensor.
            p (float): Power parameter for GeM pooling (default 3.0).

        Returns:
            Tensor: Pooled feature tensor.
        """

        return (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)

    def forward(self, x):
        """
        Forward pass of the MVL Attention module.

        Process:
        1. Spatial View (H, W): Apply attention masks on spatial dimensions.
           Reshape input to (B*T, C, H, W) -> generate masks -> pool features.
        2. Temporal-Width View (T, W): Permute input to align T and W as the last dimensions.
           Reshape to (B*H, C, T, W) -> generate masks -> restore shape -> pool features.
        3. Temporal-Height View (T, H): Permute input to align T and H as the last dimensions.
           Reshape to (B*W, C, T, H) -> generate masks -> restore shape -> pool features.
        4. Concatenate features from all heads and all views.
        5. Collect raw attention masks for diversity loss calculation.

        Args:
            x (Tensor): Input tensor of shape [B, T, C, H, W].

        Returns:
            tuple: (output, att_masks)
                - output (Tensor): Concatenated feature vectors from all views and heads.
                - att_masks (list): List of attention masks for each view [masks_hw, masks_tw, masks_th].
        """

        b, t, c, h, w = x.shape

        feats = []

        # Process spatial (h, w) masks
        x_hw = x.view(b * t, c, h, w)
        masks_hw = self.activation(self.view_attention_hw(x_hw))  # (b*t, num_heads, h, w)

        for i in range(self.num_heads):
            mask = masks_hw[:, i:i + 1, :, :]
            feat = mask * x_hw
            if self.mode == 'gem':
                feat = feat.view(b * t, c, -1)
                feat = self.gempool(feat)
            else:
                feat = F.avg_pool2d(feat, feat.size()[2:])
                feat = feat.view(feat.size(0), -1)
            feats.append(feat)

        # Process temporal (t, w) masks
        x_tw = x.permute(0, 3, 2, 1, 4).reshape(b * h, c, t, w)
        masks_tw = self.activation(self.view_attention_tw(x_tw))  # (b*h, num_heads, t, w)

        for i in range(self.num_heads):
            mask = masks_tw[:, i:i + 1, :, :]
            feat = mask * x_tw
            feat = feat.reshape(b, h, c, t, w).permute(0, 3, 2, 1, 4).reshape(b * t, c, h, w)  # Correct reshaping
            if self.mode == 'gem':
                feat = feat.view(b * t, c, -1)
                feat = self.gempool(feat)
            else:
                feat = F.avg_pool2d(feat, feat.size()[2:])
                feat = feat.view(feat.size(0), -1)
            feats.append(feat)

        # Process temporal (t, h) masks
        x_th = x.permute(0, 4, 2, 1, 3).reshape(b * w, c, t, h)
        masks_th = self.activation(self.view_attention_th(x_th))  # (b*w, num_heads, t, h)

        for i in range(self.num_heads):
            mask = masks_th[:, i:i + 1, :, :]
            feat = mask * x_th
            feat = feat.reshape(b, w, c, t, h).permute(0, 3, 2, 4, 1).reshape(b * t, c, h, w)  # Correct reshaping
            if self.mode == 'gem':
                feat = feat.view(b * t, c, -1)
                feat = self.gempool(feat)
            else:
                feat = F.avg_pool2d(feat, feat.size()[2:])
                feat = feat.view(feat.size(0), -1)
            feats.append(feat)

        # Concatenate all features
        output = torch.cat(feats, dim=1)

        att_masks = [masks_hw.view(b * t, self.num_heads, h * w),
                     masks_tw.view(b * h, self.num_heads, t * w),
                     masks_th.view(b * w, self.num_heads, t * h)]

        return output, att_masks
