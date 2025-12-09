# ------------------------------------------------------------------------------
# File:    M3-ReID/models/modules/non_local.py
#
# Description:
#    This module implements the Non-Local Neural Network block.
#    It captures long-range dependencies by computing the weighted sum of features
#    at all positions, which is crucial for modeling global context in ReID.
#
# Key Features:
# - Supports both 2D (Spatial) and 3D (Spatio-Temporal) non-local operations.
#
# Classes:
# - NonLocal
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocal(nn.Module):
    """
    Non-Local Block for capturing long-range dependencies.
    It computes the response at a position as a weighted sum of the features at all positions.
    """

    def __init__(self, in_channels, inter_ratio=4, mode='HW'):
        """
        Initialize the Non-Local block.

        Args:
            in_channels (int): Number of input channels.
            inter_ratio (int): Reduction ratio for intermediate channels (default 4).
            mode (str): Operation mode, 'HW' for 2D (Spatial) or 'THW' for 3D (Spatio-Temporal).
        """

        super(NonLocal, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = max(in_channels // inter_ratio, 1)

        if mode == 'HW':
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif mode == 'THW':
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError('mode of NonLocal should be HW or THW')

        self.theta = Conv(in_channels=self.in_channels,
                          out_channels=self.inter_channels,
                          kernel_size=1, stride=1, padding=0)
        self.phi = Conv(in_channels=self.in_channels,
                        out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
        self.g = Conv(in_channels=self.in_channels,
                      out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            Conv(in_channels=self.inter_channels,
                 out_channels=self.in_channels,
                 kernel_size=1, stride=1, padding=0),
            BatchNorm(self.in_channels),
        )

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the convolutional layers using Xavier Uniform
        and biases with constants.

        Note:
            The final BatchNorm layer in the projection block `W` is initialized with
            weights and biases set to 0.0.
            Since the Non-Local block is inserted as a residual connection (z = W(y) + x),
            this zero-initialization ensures that the module initially acts as an identity mapping.
            This is a good practice when inserting new modules into a pre-trained backbone,
            as it preserves the original pre-trained behavior at the start of training.
        """

        nn.init.xavier_uniform_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0.0)
        nn.init.xavier_uniform_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0.0)
        nn.init.xavier_uniform_(self.g.weight)
        nn.init.constant_(self.g.bias, 0.0)
        nn.init.xavier_uniform_(self.W[0].weight)
        nn.init.constant_(self.W[0].bias, 0.0)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x):
        """
        Forward pass of the Non-Local block.

        Process:
        1. Compute embedded features `theta` (query), `phi` (key), and `g` (value).
        2. Reshape and compute the pairwise affinity matrix (theta * phi).
        3. Apply Softmax to obtain the attention matrix.
        4. Compute the weighted sum of `g` features using the attention matrix.
        5. Project the result back to the original channel dimension using `W`.
        6. Add the residual connection (input `x`).

        Args:
            x (Tensor): Input tensor. Shape [B, C, H, W] for 2D or [B, C, T, H, W] for 3D.

        Returns:
            Tensor: Output tensor with captured long-range dependencies, same shape as input.
        """

        b, c, *other_dim = x.shape

        theta_x = self.theta(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        g_x = self.g(x).view(b, self.inter_channels, -1).permute(0, 2, 1)

        affinity_matrix = torch.matmul(theta_x, phi_x)
        attention_matrix = F.softmax(affinity_matrix, dim=-1)

        y = torch.matmul(attention_matrix, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, *other_dim)

        z = self.W(y) + x

        return z
