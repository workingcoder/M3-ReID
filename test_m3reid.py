# ------------------------------------------------------------------------------
# File:    M3-ReID/test_m3reid.py
#
# Description:
#    The main testing script for the M3-ReID framework.
#    It evaluates the trained model checkpoint.
#
# Key Features:
# - Loads trained weights and initializes the M3-ReID model.
# - Extracts features for Query and Gallery sets using video-level inference.
# - Computes Cosine Distance matrices for cross-modality matching.
# - Calculates and reports standard metrics: Rank-1, Rank-5, Rank-10, mAP, and mINP.
# - Evaluates both Infrared-to-Visible (I2V) and Visible-to-Infrared (V2I) modes.
#
# Paper:
#     M3-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-
#     Infrared Person Re-Identification by Liang et al.
#     See https://ieeexplore.ieee.org/document/11275868 (IEEE TIFS).
# ------------------------------------------------------------------------------

import os
import sys
import time
import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data.manager import HITSZVCMDataManager
from data.manager import BUPTCampusDataManager
from data.dataset import VideoVIDataset
from data.transform import SyncTrackTransform
from models.model_m3reid import M3ReID
from tools.eval_metrics import get_cmc_mAP_mINP
from tools.utils import time_str, Logger

if __name__ == '__main__':

    # Arguments --------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Video-based visible-infrared cross-modality ReID')

    # -- Data Arguments ------------------------------------------------------------------------------------------------
    parser.add_argument('--dataset', default='HITSZVCM', help='Dataset name')
    parser.add_argument('--dataset_dir', default='../Datasets/HITSZ-VCM', help='Directory of dataset')
    parser.add_argument('--img_h', default=288, type=int, help='Height of input images')
    parser.add_argument('--img_w', default=144, type=int, help='Width of input images')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
    parser.add_argument('--workers', default=4, type=int, help='Num of dataloader workers')

    # -- Other Arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--resume', default=None, type=str, help='Resume from path of checkpoint')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--desc', type=str, default=None, help='Description for this testing process')

    args = parser.parse_args()

    # Env  -------------------------------------------------------------------------------------------------------------
    torch.cuda.set_device(args.gpu)
    torch.set_float32_matmul_precision('high')  # highest high medium

    suffix = f'Time-{time_str()}' if (args.desc is None) else f'Time-{time_str()}_{args.desc}'

    ckptlog_dir = os.path.join('ckptlog', args.dataset, suffix)
    os.makedirs(ckptlog_dir, exist_ok=True)

    log_path = os.path.join(ckptlog_dir, f'log_{suffix}.txt')
    sys.stdout = Logger(log_path)

    modelckpt_dir = os.path.join(ckptlog_dir, 'modelckpt')
    os.makedirs(modelckpt_dir, exist_ok=True)

    print(f'Args: {args}')

    # Data -------------------------------------------------------------------------------------------------------------
    sample_seq_num = 6
    test_batch_size = args.batch_size  # Set Appropriate Values Based on GPU Memory

    # -- DataManager ---------------------------------------------------------------------------------------------------
    if args.dataset == 'HITSZVCM':
        data_manager = HITSZVCMDataManager(args.dataset_dir)
    elif args.dataset == 'BUPTCampus':
        data_manager = BUPTCampusDataManager(args.dataset_dir)
    else:
        raise RuntimeError(f'Dataset {args.dataset} is not supported for now.')

    num_train_class = data_manager.train_num_pids
    num_test_class = data_manager.test_num_pids
    num_query = len(data_manager.query_track_pids)
    num_gallery = len(data_manager.gallery_track_pids)

    # -- Dataset & Dataloader ------------------------------------------------------------------------------------------
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_test = SyncTrackTransform(T.Compose([
        T.ToPILImage(),
        T.Resize((args.img_h, args.img_w)),
        T.ToTensor(),
        normalize,
    ]))

    query_dataset = VideoVIDataset(data_manager, transform=transform_test,
                                   sample_seq_num=sample_seq_num, sample_mode='evenly', dataset_mode='query')
    gallery_dataset = VideoVIDataset(data_manager, transform=transform_test,
                                     sample_seq_num=sample_seq_num, sample_mode='evenly', dataset_mode='gallery')
    query_loader = DataLoader(query_dataset, batch_size=test_batch_size,
                              shuffle=False, pin_memory=True, num_workers=args.workers)
    gallery_loader = DataLoader(gallery_dataset, batch_size=test_batch_size,
                                shuffle=False, pin_memory=True, num_workers=args.workers)

    # Model ------------------------------------------------------------------------------------------------------------
    model = M3ReID(sample_seq_num, num_train_class).cuda()

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=True, map_location=torch.device('cuda'))
        for key in list(checkpoint.keys()):
            model_state_dict = model.state_dict()
            if key in model_state_dict:
                if torch.is_tensor(checkpoint[key]) and checkpoint[key].shape != model_state_dict[key].shape:
                    print(f'Warning during loading weights - Auto remove mismatch key: {key}')
                    checkpoint.pop(key)
        model.load_state_dict(checkpoint, strict=False)

    # Test --------------------------------------------------------------------------------------------------------
    model.eval()

    s_time = time.time()

    query_embeddings = torch.zeros((num_query, model.embedding_dim)).cuda()
    gallery_embeddings = torch.zeros((num_gallery, model.embedding_dim)).cuda()
    query_ptr, gallery_ptr = 0, 0
    q_pids, q_cids, q_mids = [], [], []
    g_pids, g_cids, g_mids = [], [], []

    with torch.no_grad():

        for track_data, pids, cids, mids in query_loader:
            inputs = track_data.cuda()
            pids = pids.cuda()
            cids = cids.cuda()
            mids = mids.cuda()
            batch_num = inputs.shape[0]
            embeddings = model(inputs)
            query_embeddings[query_ptr:query_ptr + batch_num, :] = embeddings.detach()
            query_ptr = query_ptr + batch_num
            q_pids.extend(pids)
            q_cids.extend(cids)
            q_mids.extend(mids)
        q_pids = torch.stack(q_pids, dim=0)
        q_cids = torch.stack(q_cids, dim=0)
        q_mids = torch.stack(q_mids, dim=0)

        for track_data, pids, cids, mids in gallery_loader:
            inputs = track_data.cuda()
            pids = pids.cuda()
            cids = cids.cuda()
            mids = mids.cuda()
            batch_num = inputs.shape[0]
            embeddings = model(inputs)
            gallery_embeddings[gallery_ptr:gallery_ptr + batch_num, :] = embeddings.detach()
            gallery_ptr = gallery_ptr + batch_num
            g_pids.extend(pids)
            g_cids.extend(cids)
            g_mids.extend(mids)
        g_pids = torch.stack(g_pids, dim=0)
        g_cids = torch.stack(g_cids, dim=0)
        g_mids = torch.stack(g_mids, dim=0)

    e_time_1 = time.time()

    if args.dataset == 'HITSZVCM':
        i2v_dist_mat = -torch.matmul(query_embeddings, gallery_embeddings.t())
        i2v_sorted_indices = torch.argsort(i2v_dist_mat, dim=1)
        i2v_cmc, i2v_mAP, i2v_mINP = get_cmc_mAP_mINP(i2v_sorted_indices, q_pids, q_cids, g_pids, g_cids)
        v2i_dist_mat = i2v_dist_mat.t()
        v2i_sorted_indices = torch.argsort(v2i_dist_mat, dim=1)
        v2i_cmc, v2i_mAP, v2i_mINP = get_cmc_mAP_mINP(v2i_sorted_indices, g_pids, g_cids, q_pids, q_cids)
    elif args.dataset == 'BUPTCampus':
        i2v_query_embeddings = query_embeddings[q_mids == 1]
        i2v_gallery_embeddings = gallery_embeddings[g_mids == 2]
        i2v_q_pids, i2v_q_cids = q_pids[q_mids == 1], q_cids[q_mids == 1]
        i2v_g_pids, i2v_g_cids = g_pids[g_mids == 2], g_cids[g_mids == 2]
        i2v_dist_mat = -torch.matmul(i2v_query_embeddings, i2v_gallery_embeddings.t())
        i2v_sorted_indices = torch.argsort(i2v_dist_mat, dim=1)
        i2v_cmc, i2v_mAP, i2v_mINP = get_cmc_mAP_mINP(i2v_sorted_indices,
                                                      i2v_q_pids, i2v_q_cids, i2v_g_pids, i2v_g_cids)
        v2i_query_embeddings = query_embeddings[q_mids == 2]
        v2i_gallery_embeddings = gallery_embeddings[g_mids == 1]
        v2i_q_pids, v2i_q_cids = q_pids[q_mids == 2], q_cids[q_mids == 2]
        v2i_g_pids, v2i_g_cids = g_pids[g_mids == 1], g_cids[g_mids == 1]
        v2i_dist_mat = -torch.matmul(v2i_query_embeddings, v2i_gallery_embeddings.t())
        v2i_sorted_indices = torch.argsort(v2i_dist_mat, dim=1)
        v2i_cmc, v2i_mAP, v2i_mINP = get_cmc_mAP_mINP(v2i_sorted_indices,
                                                      v2i_q_pids, v2i_q_cids, v2i_g_pids, v2i_g_cids)
    else:
        raise RuntimeError(f'Dataset {args.dataset} is not supported for now.')

    e_time_2 = time.time()

    info_str = (f'EVAL'
                f'Time: {e_time_2 - s_time:.4f} ({e_time_1 - s_time:.4f} + {e_time_2 - e_time_1:.4f})s \n'
                f'Mode - i2v  '
                f'r1: {i2v_cmc[0]:.2%} '
                f'r5: {i2v_cmc[4]:.2%} '
                f'r10: {i2v_cmc[9]:.2%} '
                f'r20: {i2v_cmc[19]:.2%} '
                f'mAP: {i2v_mAP:.2%} '
                f'mINP: {i2v_mINP:.2%} \n'
                f'Mode - v2i  '
                f'r1: {v2i_cmc[0]:.2%} '
                f'r5: {v2i_cmc[4]:.2%} '
                f'r10: {v2i_cmc[9]:.2%} '
                f'r20: {v2i_cmc[19]:.2%} '
                f'mAP: {v2i_mAP:.2%} '
                f'mINP: {v2i_mINP:.2%} ')

    info_str = '~' * 100 + '\n' + info_str + '\n' + '~' * 100
    print(info_str)

    # Save -------------------------------------------------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(modelckpt_dir, f'model.pth'))
