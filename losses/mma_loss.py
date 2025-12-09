# ------------------------------------------------------------------------------
# File:    M3-ReID/losses/mma_loss.py
#
# Description:
#    This module implements the Multi-Modality Alignment (MMA) Loss.
#    It is designed to bridge the distribution gap between Visible and Infrared
#    modalities by unifying metric learning with cross-modality retrieval objectives.
#
# Key Features:
# - Simulates the cross-modality retrieval process during training.
# - Explicitly aligns embeddings by ensuring same-identity cross-modal pairs
#   are ranked higher than different-identity pairs.
# - Uses Soft Margin Loss to optimize the distance gap.
#
# Classes:
# - MultiModalityAlignmentLoss
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiModalityAlignmentLoss(nn.Module):
    """
    Implements the Multi-Modality Alignment (MMA) loss of M3-ReID.
    This loss function organizes the training batch into query (one modality) and
    gallery (other modalities) sets to strictly enforce cross-modality matching constraints.

    'M3-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-Infrared Person Re-Identification'
    by Liang et al. See https://ieeexplore.ieee.org/document/11275868 (IEEE TIFS).
    """

    def __init__(self):
        """
        Initialize the MMA Loss module.
        """

        super(MultiModalityAlignmentLoss, self).__init__()

    def forward(self, embeddings, id_labels, m_labels):
        """
        Computes the MMA loss given a batch of embeddings and labels.

        Process:
        1. Group embeddings and labels by their modality (e.g., IR vs. RGB).
        2. Iterate through each modality, treating it as the 'query' set and others as the 'gallery'.
        3. Construct boolean masks to identify positive (same ID) and negative (diff ID) cross-modal pairs.
        4. Filter out invalid queries (those with no matching positives or negatives in the current batch).
        5. Compute the Cosine Distance between query and gallery embeddings.
        6. Calculate the average distance for positive pairs (`match_dist`) and negative pairs (`mismatch_dist`).
        7. Apply Soft Margin Loss to maximize `(mismatch_dist - match_dist)`.

        Args:
            embeddings (Tensor): The feature vectors extracted from the model [Batch, Dim].
            id_labels (Tensor): The identity labels for the batch [Batch].
            m_labels (Tensor): The modality labels for the batch [Batch].

        Returns:
            Tensor: The computed scalar loss value.
        """

        m_labels_unique = torch.unique(m_labels)
        m_num = len(m_labels_unique)

        embeddings_list = [embeddings[m_labels == m_label] for m_label in m_labels_unique]
        id_labels_list = [id_labels[m_labels == m_label] for m_label in m_labels_unique]

        mma_loss = 0
        valid_m_count = 0
        for i in range(m_num):
            cur_m_embeddings = embeddings_list[i]
            cur_m_id_labels = id_labels_list[i]
            other_m_embeddings = torch.cat([embeddings_list[j] for j in range(len(m_labels_unique)) if j != i], dim=0)
            other_m_id_labels = torch.cat([id_labels_list[j] for j in range(len(m_labels_unique)) if j != i], dim=0)

            match_mask = cur_m_id_labels.unsqueeze(dim=1) == other_m_id_labels.unsqueeze(dim=0)
            mismatch_mask = ~match_mask
            match_num = match_mask.sum(dim=-1)
            mismatch_num = mismatch_mask.sum(dim=-1)

            # remove invalid queries
            # (It has no cross-modal matching in the batch, or all cross-modal matching is the same class)
            remove_mask = (match_num == 0) | (match_num == len(other_m_id_labels))
            if remove_mask.all():
                continue
            cur_m_embeddings = cur_m_embeddings[~remove_mask]
            cur_m_id_labels = cur_m_id_labels[~remove_mask]
            match_mask = match_mask[~remove_mask]
            mismatch_mask = mismatch_mask[~remove_mask]
            match_num = match_num[~remove_mask]
            mismatch_num = mismatch_num[~remove_mask]

            dist_mat = F.cosine_similarity(cur_m_embeddings[:, None, :], other_m_embeddings[None, :, :], dim=-1)
            dist_mat = (1 - dist_mat) / 2

            match_dist_mat = dist_mat * match_mask
            mismatch_dist_mat = dist_mat * mismatch_mask

            # imp1 - all pairs
            match_dist = match_dist_mat.sum(dim=-1) / match_num
            mismatch_dist = mismatch_dist_mat.sum(dim=-1) / mismatch_num
            # imp2 - hardest pairs [Optional]
            # match_dist, _ = match_dist_mat.max(dim=-1)
            # mismatch_dist, _ = mismatch_dist_mat.min(dim=-1)

            mma_loss += F.soft_margin_loss(mismatch_dist - match_dist, torch.ones_like(match_dist))
            valid_m_count += 1

        mma_loss = mma_loss / valid_m_count if valid_m_count > 0 else 0

        return mma_loss
