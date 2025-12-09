# ------------------------------------------------------------------------------
# File:    M3-ReID/data/dataset.py
#
# Description:
#     This module defines the dataset handling logic for the M3-ReID framework.
#
# Key Features:
# - Loads video tracks for Training, Query, and Gallery sets.
# - Implements specific sampling strategies tailored for VVI-ReID.
#
# Classes:
# - VideoVIDataset: Main class for iterating over video track data.
# ------------------------------------------------------------------------------

import math
import random
import numpy as np
from torch.utils.data import Dataset

from tools.utils import read_image


class VideoVIDataset(Dataset):
    """
    A custom Dataset class for loading video tracks from VVI-ReID datasets (e.g., BUPTCampus, HITSZ-VCM).
    It manages video sequence retrieval with specific sampling strategies to handle temporal dynamics.
    """

    def __init__(self, data_manager, transform, sample_seq_num, sample_mode='evenly', dataset_mode='train'):
        """
        Initialize the dataset by extracting tracks, PIDs, camera IDs, and modality IDs
        from the data_manager.

        Args:
            data_manager: Object containing the raw dataset paths and metadata.
            transform (callable or tuple): Transformations to apply to the images.
                                           Can be a tuple (ir_trans, vis_trans) for specific modalities.
            sample_seq_num (int): The target sequence length for the model input.
            sample_mode (str): The strategy to sample frames ('evenly', 'random', 'all').
            dataset_mode (str): The mode of the dataset ('train', 'query', 'gallery').
        """

        super(VideoVIDataset, self).__init__()
        assert sample_mode in ['evenly', 'random', 'all']
        assert dataset_mode in ['train', 'query', 'gallery']

        if dataset_mode == 'train':
            self.tracks = data_manager.train_tracks
            self.pids = data_manager.train_track_pids
            self.cids = data_manager.train_track_cids
            self.mids = data_manager.train_track_mids
            self.num_unique_pids = data_manager.train_num_pids
        elif dataset_mode == 'query':
            self.tracks = data_manager.query_tracks
            self.pids = data_manager.query_track_pids
            self.cids = data_manager.query_track_cids
            self.mids = data_manager.query_track_mids
            self.num_unique_pids = data_manager.test_num_pids
        else:  # dataset_mode == 'gallery':
            self.tracks = data_manager.gallery_tracks
            self.pids = data_manager.gallery_track_pids
            self.cids = data_manager.gallery_track_cids
            self.mids = data_manager.gallery_track_mids
            self.num_unique_pids = data_manager.test_num_pids

        self.transform = transform
        self.sample_seq_num = sample_seq_num
        self.sample_mode = sample_mode
        self.dataset_mode = dataset_mode

    def __len__(self):
        """
        Returns the total number of video tracks in the current dataset.
        """

        return len(self.tracks)

    def __getitem__(self, index):
        """
        Retrieves a video clip and its metadata at the specified index.

        Process:
        1. Identify the track and modality (Visible or Infrared) to select the correct transform.
        2. Pad the video sequence if it is shorter than the required length.
        3. Sample indices based on the 'sample_mode' (evenly/random) to form the clip.
        4. Load images, apply transforms, and stack them into a tensor.

        Args:
            index (int): Index of the video track to retrieve.

        Returns:
            tuple: (track_data, track_pid, track_cid, track_mid)
                - track_data (Tensor): The processed video clip tensor.
                - track_pid (int): Person ID.
                - track_cid (int): Camera ID.
                - track_mid (int): Modality ID (1 for IR, 2 for RGB).
        """

        track_img_paths = self.tracks[index]
        track_pid = self.pids[index]
        track_cid = self.cids[index]
        track_mid = self.mids[index]
        cur_seq_num = len(track_img_paths)
        frame_indices = list(range(cur_seq_num))

        if type(self.transform) == tuple:
            transform = self.transform[track_mid - 1]  # track_mid: 1 for ir, 2 for rgb - Attention !!!!!
        else:
            transform = self.transform

        if cur_seq_num < self.sample_seq_num:
            supplement_num = self.sample_seq_num - cur_seq_num
            indices = frame_indices + [frame_indices[-1]] * supplement_num  # Compensating with last frame
            ## Different compensating strategies [Optional]
            # indices = (frame_indices + frame_indices * (supplement_num // cur_seq_num) +
            #            frame_indices[:supplement_num % cur_seq_num])  # Compensating with repetition from first frame
        else:
            supplement_num = math.ceil(cur_seq_num / self.sample_seq_num) * self.sample_seq_num - cur_seq_num
            indices = frame_indices + [frame_indices[-1]] * supplement_num  # Compensating with last frame
            ## Different compensating strategies [Optional]
            # indices = frame_indices + frame_indices[:supplement_num]  # Compensating with repetition from first frame
        indices_num = len(indices)
        tracks_num = int(indices_num / self.sample_seq_num)

        if self.sample_mode == 'evenly':
            # Evenly sample sample_seq_num consecutive frames from cur_seq_num frames
            if self.dataset_mode == 'train':
                k = [i * tracks_num + random.randint(0, tracks_num - 1) for i in range(self.sample_seq_num)]
                sample_indices = list(np.array(indices)[k])
                ## Different sampling strategies [Optional]
                # k = random.randint(0, tracks_num - 1)
                # sample_indices = [idx for i, idx in enumerate(indices) if i % tracks_num == k]
            else:  # self.dataset_mode in ['query', 'gallery']
                k = 0
                sample_indices = [idx for i, idx in enumerate(indices) if i % tracks_num == k]
            track_data = []
            for idx in sample_indices:
                img_path = track_img_paths[idx]
                img = read_image(img_path)
                track_data.append(img)
            track_data = transform(track_data)

            return track_data, track_pid, track_cid, track_mid

        if self.sample_mode == 'random':
            # Randomly sample sample_seq_num consecutive frames from cur_seq_num frames
            begin_index = random.randint(0, indices_num - self.sample_seq_num)
            end_index = begin_index + self.sample_seq_num
            sample_indices = indices[begin_index:end_index]
            track_data = []
            for idx in sample_indices:
                img_path = track_img_paths[idx]
                img = read_image(img_path)
                track_data.append(img)
            track_data = transform(track_data)

            return track_data, track_pid, track_cid, track_mid

        if self.sample_mode == 'all':
            # Sample all frames in a video into a list of tracks (each track contains sample_seq_num frames)
            track_data_list = []
            for i in range(tracks_num):
                track_data = []
                for j in range(self.sample_seq_num):
                    # img_path = img_paths[indices[i * self.sample_seq_num + j]]  # dense
                    img_path = track_img_paths[indices[j * tracks_num + i]]  # even
                    img = read_image(img_path)
                    track_data.append(img)
                track_data = transform(track_data)
                track_data_list.append(track_data)

            return track_data_list, track_pid, track_cid, track_mid

        else:
            raise KeyError(
                "Unknown sample_mode method: {}. Expected one of {}".format(self.sample_mode, self.sample_methods))
