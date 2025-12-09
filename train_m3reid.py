# ------------------------------------------------------------------------------
# File:    M3-ReID/train_m3reid.py
#
# Description:
#    The main training script for the M3-ReID framework.
#    It orchestrates the entire pipeline including data loading, model initialization,
#    loss computation (ID, MMA, OFR, DAC), optimization, and evaluation.
#
# Key Features:
# - Supports training on HITSZ-VCM and BUPTCampus datasets.
# - Implements the full M3-ReID training loop with mixed precision (AMP) support.
# - Handles evaluation on both Visible-to-Infrared and Infrared-to-Visible modes.
# - Logs metrics to TensorBoard and text files.
# - Save checkpoints based on specified intervals.
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
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.manager import HITSZVCMDataManager
from data.manager import BUPTCampusDataManager
from data.sampler import NormTripletSampler
from data.sampler import CrossModalityTripletSampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import CrossModalityIdentitySampler
from data.sampler import IdentityCrossModalitySampler
from data.dataset import VideoVIDataset
from data.transform import SyncTrackTransform
from data.transform import WeightedGrayscale
from data.transform import StyleVariation
from models.model_m3reid import M3ReID
from losses.mma_loss import MultiModalityAlignmentLoss
from losses.sep_loss import SeparationLoss
from tools.eval_metrics import get_cmc_mAP_mINP
from tools.utils import set_seed, time_str, Logger

if __name__ == '__main__':

    # Arguments --------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Video-based visible-infrared cross-modality ReID')

    # -- Data Arguments ------------------------------------------------------------------------------------------------
    parser.add_argument('--dataset', default='HITSZVCM', help='Dataset name')
    parser.add_argument('--dataset_dir', default='../Datasets/HITSZ-VCM', help='Directory of dataset')
    parser.add_argument('--img_h', default=288, type=int, help='Height of input images')
    parser.add_argument('--img_w', default=144, type=int, help='Width of input images')
    parser.add_argument('--p_num', default=4, type=int, help='Num of identities')
    parser.add_argument('--k_num', default=8, type=int, help='Num of samples per identity')
    parser.add_argument('--workers', default=4, type=int, help='Num of dataloader workers')

    # -- Optim Arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate for adam optimizer')
    parser.add_argument('--wd', default=0.0005, type=float, help='Weight decay for adam optimizer')

    # -- Other Arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--fp16', action='store_true', default=False, help='Whether to use AMP')
    parser.add_argument('--resume', default=None, type=str, help='Resume from path of checkpoint')

    parser.add_argument('--log_interval', default=10, type=int, help='Interval of logging')
    parser.add_argument('--test_interval', default=1, type=int, help='Interval of testing')
    parser.add_argument('--save_interval', default=1, type=int, help='Interval of saving checkpoints')

    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--desc', type=str, default=None, help='Description for this training process')

    args = parser.parse_args()

    # Env  -------------------------------------------------------------------------------------------------------------
    torch.cuda.set_device(args.gpu)
    torch.set_float32_matmul_precision('high')  # highest high medium

    set_seed(args.seed)
    suffix = f'Time-{time_str()}' if (args.desc is None) else f'Time-{time_str()}_{args.desc}'

    ckptlog_dir = os.path.join('ckptlog', args.dataset, suffix)
    os.makedirs(ckptlog_dir, exist_ok=True)

    tensorboard_dir = os.path.join(ckptlog_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    log_path = os.path.join(ckptlog_dir, f'log_{suffix}.txt')
    sys.stdout = Logger(log_path)

    modelckpt_dir = os.path.join(ckptlog_dir, 'modelckpt')
    os.makedirs(modelckpt_dir, exist_ok=True)

    print(f'Args: {args}')

    # Data -------------------------------------------------------------------------------------------------------------
    sample_seq_num = 6
    train_batch_size = args.p_num * args.k_num
    test_batch_size = train_batch_size  # Set Appropriate Values Based on GPU Memory

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

    transform_train_ir = SyncTrackTransform(T.Compose([
        T.ToPILImage(),
        T.Resize((args.img_h, args.img_w)),
        T.RandomCrop((args.img_h, args.img_w), padding=5, fill=0),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
        T.RandomErasing(),
        StyleVariation(mode='one', p=1.0),
    ]))
    transform_train_rgb = SyncTrackTransform(T.Compose([
        T.ToPILImage(),
        T.Resize((args.img_h, args.img_w)),
        WeightedGrayscale(p=0.5),
        T.RandomCrop((args.img_h, args.img_w), padding=5, fill=0),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
        T.RandomErasing(),
        StyleVariation(mode='all', p=1.0),
    ]))
    transform_train = (transform_train_ir, transform_train_rgb)

    transform_test = SyncTrackTransform(T.Compose([
        T.ToPILImage(),
        T.Resize((args.img_h, args.img_w)),
        T.ToTensor(),
        normalize,
    ]))

    train_loader = None  # Get the train_loader in Each Epoch of Training for Random Selection

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

    # Loss -------------------------------------------------------------------------------------------------------------
    criterion_ce_loss = nn.CrossEntropyLoss().cuda()
    criterion_mma_loss = MultiModalityAlignmentLoss().cuda()
    criterion_ofr_loss = SeparationLoss().cuda()
    criterion_dac_loss = SeparationLoss().cuda()

    # Optimizer --------------------------------------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[80, 120], gamma=0.1)

    # Iteration --------------------------------------------------------------------------------------------------------
    total_epoch_num = 200
    sample_method = 'norm_triplet'

    if args.fp16: amp_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(total_epoch_num):
        train_dataset = VideoVIDataset(data_manager, transform=transform_train,
                                       sample_seq_num=sample_seq_num, sample_mode='evenly', dataset_mode='train')

        shuffle = False
        if sample_method == 'norm_triplet':
            sampler = NormTripletSampler(train_dataset, train_batch_size, args.k_num)
        elif sample_method == 'cross_modality_triplet':
            ratio = 0.5  # Set to a custom value
            sampler = CrossModalityTripletSampler(train_dataset, train_batch_size, args.k_num, modal_ratio=ratio)
        elif sample_method == 'cross_modality_random':
            sampler = CrossModalityRandomSampler(train_dataset, train_batch_size)
        elif sample_method == 'cross_modality_identity':
            sampler = CrossModalityIdentitySampler(train_dataset, args.p_num, args.k_num)
        elif sample_method == 'identity_cross_modality':
            sampler = IdentityCrossModalitySampler(train_dataset, train_batch_size, args.k_num)
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(train_dataset, train_batch_size, sampler=sampler,
                                  shuffle=shuffle, drop_last=True, pin_memory=True, num_workers=args.workers)

        # -- Train -----------------------------------------------------------------------------------------------------
        model.train()
        s_time = time.time()
        for batch_idx, batch_data in enumerate(train_loader):
            track_data, track_pid, track_cid, track_mid = batch_data
            inputs, labels = track_data.cuda(), track_pid.cuda()

            with torch.amp.autocast(device_type='cuda', enabled=args.fp16):
                x_embed, x_embed_m, x_logits, x_logits_m, mvl_att_masks = model(inputs)
                id_labels = labels
                m_labels = track_mid.cuda()

                loss_ofr = criterion_ofr_loss(x_embed)

                loss_dac = sum([criterion_dac_loss(mvl_att_masks[i]) for i in range(len(mvl_att_masks))])

                loss_mma = criterion_mma_loss(x_embed_m, id_labels, m_labels)
                loss_id = criterion_ce_loss(x_logits_m, labels)

                b, t, c = x_embed.shape
                id_labels_all = id_labels.repeat_interleave(t)
                m_labels_all = m_labels.repeat_interleave(t)
                loss_mma_frames = criterion_mma_loss(x_embed.view(b * t, c), id_labels_all, m_labels_all)
                loss_id_frames = criterion_ce_loss(x_logits.view(b * t, -1), id_labels_all)

            _, predicted = x_logits_m.max(dim=1)
            cls_acc = (predicted.eq(labels).sum().item()) / len(labels)

            loss_mid = loss_id + loss_id_frames
            loss_mma = loss_mma + loss_mma_frames

            loss = loss_mid + loss_mma + loss_ofr + loss_dac

            optimizer.zero_grad()
            if args.fp16:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']

            if (batch_idx + 1) % args.log_interval == 0:
                batch_num = len(train_loader)
                iter_num = epoch * batch_num + batch_idx + 1
                e_time = time.time()
                print(f'Epoch: [{epoch + 1}][{batch_idx + 1}/{batch_num}] '
                      f'Time: {e_time - s_time:.4f}s '
                      f'lr:{current_lr:.8f} '
                      f'cls_acc: {cls_acc:.4f} '
                      f'loss_mid: {loss_mid.data:.4f} '
                      f'loss_mma: {loss_mma.data:.4f} '
                      f'loss_ofr: {loss_ofr.data:.4f} '
                      f'loss_dac: {loss_dac.data:.4f} '
                      )
                s_time = time.time()
                writer.add_scalar('metric/cls_acc', cls_acc, iter_num)
                writer.add_scalar('metric/loss_mid', loss_mid.data, iter_num)
                writer.add_scalar('metric/loss_mma', loss_mma.data, iter_num)
                writer.add_scalar('metric/loss_ofr', loss_ofr.data, iter_num)
                writer.add_scalar('metric/loss_dac', loss_dac.data, iter_num)

        lr_scheduler.step()

        if epoch % args.test_interval == 0:
            # -- Test --------------------------------------------------------------------------------------------------
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

            info_str = (f'EVAL - Epoch: [{epoch + 1}] '
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

            # Log Overall Performance
            writer.add_scalar('eval/i2v_r1', i2v_cmc[0], epoch + 1)
            writer.add_scalar('eval/i2v_r5', i2v_cmc[4], epoch + 1)
            writer.add_scalar('eval/i2v_r10', i2v_cmc[9], epoch + 1)
            writer.add_scalar('eval/i2v_r20', i2v_cmc[19], epoch + 1)
            writer.add_scalar('eval/i2v_mAP', i2v_mAP, epoch + 1)
            writer.add_scalar('eval/i2v_mINP', i2v_mINP, epoch + 1)
            writer.add_scalar('eval/v2i_r1', v2i_cmc[0], epoch + 1)
            writer.add_scalar('eval/v2i_r5', v2i_cmc[4], epoch + 1)
            writer.add_scalar('eval/v2i_r10', v2i_cmc[9], epoch + 1)
            writer.add_scalar('eval/v2i_r20', v2i_cmc[19], epoch + 1)
            writer.add_scalar('eval/v2i_mAP', v2i_mAP, epoch + 1)
            writer.add_scalar('eval/v2i_mINP', v2i_mINP, epoch + 1)

        if epoch % args.save_interval == 0:
            # -- Save --------------------------------------------------------------------------------------------------
            torch.save(model.state_dict(), os.path.join(modelckpt_dir, f'model_epoch-{epoch + 1}.pth'))

    writer.close()
