# ------------------------------------------------------------------------------
# File:    M3-ReID/tools/eval_metrics.py
#
# Description:
#    This module implements standard evaluation metrics for Person Re-Identification.
#    It calculates Cumulative Matching Characteristics (CMC), mean Average Precision (mAP),
#    and mean Inverse Negative Penalty (mINP) based on the retrieval ranking results.
#
# Key Features:
# - Supports standard ReID evaluation protocol (excluding same-camera matches).
# - Computes CMC curve (Rank-1, Rank-5, etc.).
# - Computes mAP for overall retrieval quality.
# - Computes mINP to measure the cost of retrieving the hardest positive sample.
# - Fully GPU-based implementation for efficient large-scale evaluation.
#
# Main Functions:
# - get_cmc_mAP_mINP
# ------------------------------------------------------------------------------

import torch


def get_cmc_mAP_mINP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, cmc_slice=20):
    """
    Computes CMC, mAP, and mINP metrics for the retrieval results.

    Process:
    1. Iterate through each query sample.
    2. Retrieve the sorted gallery indices corresponding to the query.
    3. Filter Invalid Matches: Remove gallery samples that share the same identity AND same camera as the query
       (standard ReID protocol to avoid trivial matches).
    4. Compute Matches: Identify positive matches (same identity) in the remaining ranked list.
    5. If valid matches exist:
       - CMC: Calculate cumulative sum of matches to determine if a correct match appears within top-k ranks.
       - AP: Compute Average Precision based on the positions of positive matches.
       - INP: Compute Inverse Negative Penalty based on the position of the hardest (last) positive match.
    6. Average the metrics over all valid queries.

    Args:
        sorted_indices (Tensor): Ranking indices of gallery samples for each query [Num_Query, Num_Gallery].
        query_ids (Tensor): Identity labels for query samples [Num_Query].
        query_cam_ids (Tensor): Camera labels for query samples [Num_Query].
        gallery_ids (Tensor): Identity labels for gallery samples [Num_Gallery].
        gallery_cam_ids (Tensor): Camera labels for gallery samples [Num_Gallery].
        cmc_slice (int): The maximum rank to compute for the CMC curve (default 20).

    Returns:
        tuple: (CMC, mAP, mINP)
            - CMC (Tensor): Cumulative Matching Characteristic curve values (e.g., Rank-1, Rank-5).
            - mAP (float): Mean Average Precision.
            - mINP (float): Mean Inverse Negative Penalty.
    """

    match_id_result = gallery_ids[sorted_indices]
    match_cam_result = gallery_cam_ids[sorted_indices]

    CMC_all = []
    AP_sum = 0
    INP_sum = 0
    valid_query_num = 0

    for query_index in range(sorted_indices.shape[0]):
        result_i = match_id_result[query_index, :]

        # remove gallery samples from the same camera of the query
        same_cam_query_mask = (match_cam_result[query_index, :] == query_cam_ids[query_index]) & (
                query_ids[query_index] == result_i)
        result_i = result_i[~same_cam_query_mask]

        # match for query i
        match_mask = result_i == query_ids[query_index]
        match_num = torch.sum(match_mask)

        # if there is true matching in gallery
        if match_num != 0:
            valid_query_num += 1
            # CMC
            CMC = match_mask.cumsum(dim=0)
            CMC[CMC > 1] = 1
            CMC_all.append(CMC[:cmc_slice])
            # AP
            match_rank = torch.nonzero(match_mask).squeeze(dim=-1)
            ap = torch.mean((torch.arange(1, match_num + 1, device=sorted_indices.device) / (match_rank + 1)))
            AP_sum += ap
            # INP
            hardest_match_pos = match_rank[-1] + 1
            NP = (hardest_match_pos - match_num) / hardest_match_pos
            INP = 1 - NP
            INP_sum += INP

    CMC_all = torch.stack(CMC_all)
    CMC = CMC_all.sum(dim=0) / valid_query_num
    mAP = AP_sum / valid_query_num
    mINP = INP_sum / valid_query_num
    return CMC, mAP, mINP
