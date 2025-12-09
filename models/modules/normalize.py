# ------------------------------------------------------------------------------
# File:    M3-ReID/models/modules/normalize.py
#
# Description:
#    This module implements a standard Lp-normalization layer.
#    It is typically used to normalize feature embeddings before metric learning losses
#    (like Triplet Loss or Contrastive Loss) or during inference for cosine similarity.
#
# Key Features:
# - Configurable power (p) for Lp-norm (default L2).
# - Numerical stability handling via dimension-keeping.
#
# Classes:
# - Normalize
# ------------------------------------------------------------------------------

import torch.nn as nn


class Normalize(nn.Module):
    """
    Performs Lp normalization on input vectors.
    Usually applied to the final feature embeddings to project them onto a hypersphere.
    """

    def __init__(self, power=2):
        """
        Initialize the Normalize layer.

        Args:
            power (int): The exponent value P for Lp normalization (default 2 for L2-norm).
        """

        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        """
        Applies normalization to the input tensor.

        Process:
        1. Compute the norm of vector x along the last dimension: ||x||_p.
        2. Divide x by its norm to obtain unit vectors.

        Args:
            x (Tensor): Input tensor of shape [Batch, ..., Dim].

        Returns:
            Tensor: Normalized tensor with the same shape as input.
        """

        norm = x.pow(self.power).sum(-1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
