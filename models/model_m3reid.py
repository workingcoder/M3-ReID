# ------------------------------------------------------------------------------
# File:    M3-ReID/models/model_m3reid.py
#
# Description:
#    This module defines the complete M3-ReID network architecture.
#    It integrates the ResNet-50 backbone with Spatio-temporal Non-Local blocks
#    and the Multi-View Learning (MVL) module to extract robust video representations.
#
# Key Features:
# - ResNet-50 Backbone with configurable Non-Local block insertion.
# - Integration of Multi-View Learning Attention for spatio-temporal feature mining.
# - Multi-Granularity feature extraction (Frame-level and Video-level).
#
# Classes:
# - M3ReID
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from models.backbones.resnet import resnet50
from models.modules.non_local import NonLocal
from models.modules.mvl_attention import MultiViewLearningAttention
from models.modules.normalize import Normalize


class M3ReID(nn.Module):
    """
    The main M3-ReID model architecture.
    It unifies Multi-View, Multi-Granularity, and Multi-Modality components into
    an end-to-end trainable framework.

    'M3-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-Infrared Person Re-Identification'
    by Liang et al. See https://ieeexplore.ieee.org/document/11275868 (IEEE TIFS).
    """

    def __init__(self, sample_seq_num, class_num):
        """
        Initialize the M3-ReID model.

        Process:
        1. Load pre-trained ResNet-50 backbone.
        2. Define indices for inserting Non-Local blocks into ResNet layers.
        3. Initialize Non-Local modules for intermediate layers (mainly Layer 2 and 3).
        4. Initialize the Multi-View Learning (MVL) attention module.
        5. Setup the BNNeck (BatchNorm1d).
        6. Setup the Classification heads (Linear layers) for both frame-level and video-level supervision.

        Args:
            sample_seq_num (int): Number of frames in input video clips.
            class_num (int): Number of identity classes for classification.
        """

        super(M3ReID, self).__init__()

        self.embedding_dim = 2048  # ResNet
        self.sample_seq_num = sample_seq_num
        self.class_num = class_num

        self.backbone = resnet50(pretrained=True)

        layers = [3, 4, 6, 3]
        non_layers = [0, 2, 3, 0]
        self.NL_1 = nn.ModuleList(
            [NonLocal(256, mode='THW') for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [NonLocal(512, mode='THW') for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [NonLocal(1024, mode='THW') for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [NonLocal(2048, mode='THW') for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        num_heads = 2
        self.mvl_attention = MultiViewLearningAttention(self.embedding_dim, num_heads=num_heads, mode='gem')

        self.embedding_dim = self.embedding_dim * (num_heads * 3)

        self.bn_neck = nn.BatchNorm1d(self.embedding_dim)
        nn.init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)

        self.classifier_frame = nn.Linear(self.embedding_dim, class_num, bias=False)
        self.classifier = nn.Linear(self.embedding_dim, class_num, bias=False)

        self.l2_norm = Normalize(power=2)

    def forward(self, inputs):
        """
        Forward pass of the M3-ReID network.

        Process:
        1. Stem: Reshape input [B, T, C, H, W] to [B*T, C, H, W] and pass through ResNet stem.
        2. Backbone & Non-Local: Iterate through ResNet layers. At specific indices, reshape features
           to include the temporal dimension [B, T, C, H, W], apply the Spatiotemporal Non-Local block,
           and reshape back to [B*T, C, H, W].
        3. Multi-View Learning: Pass global features to the MVL module to obtain multi-view aggregated
           embeddings and attention masks.
        4. Bottleneck: Apply BNNeck to the embeddings.
        5. Multi-Granularity:
           - Frame-level: Keep features as [B, T, C].
           - Video-level: Average pool frame features across time to get [B, C].
        6. Heads: Compute logits for both frame and video representations.

        Args:
            inputs (Tensor): Input video tensor of shape [B, T, C, H, W].

        Returns:
            If training:
                tuple: (x_embed, x_embed_mean, x_logits, x_logits_mean, mvl_att_masks)
            If evaluating:
                Tensor: Normalized video-level embedding [B, C].
        """

        b, t, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)

        inputs = self.backbone.conv1(inputs)
        inputs = self.backbone.bn1(inputs)
        inputs = self.backbone.relu(inputs)
        inputs = self.backbone.maxpool(inputs)

        x = inputs
        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.backbone.layer1)):
            x = self.backbone.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = x.view(b, t, C, H, W).permute(0, 2, 1, 3, 4)
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.backbone.layer2)):
            x = self.backbone.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = x.view(b, t, C, H, W).permute(0, 2, 1, 3, 4)
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.backbone.layer3)):
            x = self.backbone.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = x.view(b, t, C, H, W).permute(0, 2, 1, 3, 4)
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.backbone.layer4)):
            x = self.backbone.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = x.view(b, t, C, H, W).permute(0, 2, 1, 3, 4)
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
                x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        global_feat = x

        _, C, H, W = global_feat.shape
        global_feat = global_feat.view(b, t, C, H, W)
        x_pool, mvl_att_masks = self.mvl_attention(global_feat)

        x_embed = self.bn_neck(x_pool)

        b, c = x_pool.shape
        x_pool = x_pool.view(-1, t, c)
        x_pool_mean = torch.mean(x_pool, dim=1)
        x_embed = x_embed.view(-1, t, c)
        x_embed_mean = torch.mean(x_embed, dim=1)

        if self.training:
            b, t, c = x_embed.shape
            x_logits = self.classifier_frame(x_embed.view(b * t, c)).view(b, t, -1)
            x_logits_mean = self.classifier(x_embed_mean)
            return x_embed, x_embed_mean, x_logits, x_logits_mean, mvl_att_masks
        else:
            return self.l2_norm(x_embed_mean)
