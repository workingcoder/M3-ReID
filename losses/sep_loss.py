# ------------------------------------------------------------------------------
# File:    M3-ReID/losses/sep_loss.py
#
# Description:
#    This module implements a generic Separation Loss designed to enforce
#    orthogonality or diversity among a set of feature vectors.
#
# Key Features:
# - Implements the Orthogonal Frame Regularizer (OFR) to reduce frame redundancy.
# - Implements the Diverse Attention Constraint (DAC) to encourage multi-head diversity.
# - Uses cosine similarity and MSE to minimize off-diagonal correlations.
#
# Classes:
# - SeparationLoss
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparationLoss(nn.Module):
    """
    Computes a separation loss to decorrelate vectors within a set.

    For brevity, it's a unified implementation for two losses of M3-ReID:
    1. Diverse Attention Constraint (DAC): Enforces diversity among attention heads in the Multi-View module.
    2. Orthogonal Frame Regularizer (OFR): Minimizes mutual information among frame-level embeddings.

    'M3-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-Infrared Person Re-Identification'
    by Liang et al. See https://ieeexplore.ieee.org/document/11275868 (IEEE TIFS).
    """

    def __init__(self):
        """
        Initialize the Separation Loss module.
        """

        super(SeparationLoss, self).__init__()

    def forward(self, x):
        """
        Calculates the Mean Squared Error between the off-diagonal elements of the
        cosine similarity matrix and zero, effectively forcing the vectors to be orthogonal.

        Process:
        1. Check input dimensions (if K=1, separation is impossible/unnecessary, return 0).
        2. Compute the pairwise Cosine Similarity matrix between the K vectors.
        3. Create a mask to exclude diagonal elements (self-similarity).
        4. Extract off-diagonal elements.
        5. Compute MSE loss against a target of 0 (orthogonal).

        Args:
            x (Tensor): Input tensor of shape [Batch, K, Dim], where K represents
                        the number of vectors to separate (e.g., Number of Heads or Number of Frames).

        Returns:
            Tensor: The computed scalar loss value.
        """

        b, k, c = x.shape

        if k == 1:
            return torch.tensor(0, device=x.device, dtype=x.dtype)

        similarity_matrix = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)
        mask = torch.eye(k, device=x.device).bool()
        similarity_matrix_no_diag = similarity_matrix.masked_select(~mask)
        loss = F.mse_loss(similarity_matrix_no_diag, torch.zeros_like(similarity_matrix_no_diag))

        return loss
