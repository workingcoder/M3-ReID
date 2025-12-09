# ------------------------------------------------------------------------------
# File:    M3-ReID/data/manager.py
#
# Description:
#    This module defines the Data Managers for parsing raw VVI-ReID datasets.
#    It handles directory traversal, metadata parsing, and train/test splitting
#    for specific benchmarks.
#
# Key Features:
# - Supports HITSZ-VCM and BUPTCampus datasets.
# - Parses complex file naming conventions and JSON metadata.
# - Manages ID relabeling for training and Train/Query/Gallery splitting.
# - Calculates and prints detailed dataset statistics.
#
# Classes:
# - HITSZVCMDataManager: Manager for the HITSZ-VCM dataset.
# - BUPTCampusDataManager: Manager for the BUPTCampus dataset.
# ------------------------------------------------------------------------------

import os
import json
import numpy as np


class HITSZVCMDataManager:
    """
    Data Manager for the HITSZ-VCM dataset.
    It parses the specific folder structure and filename convention
    to extract identities, cameras, and modalities.
    """

    def __init__(self, dataset_dir, min_seq_len=12):
        """
        Initialize the HITSZ-VCM data manager.

        Process:
        1. Verify dataset paths.
        2. Load track metadata (start/end indices, PIDs).
        3. Split Test set into Query and Gallery based on provided indices.
        4. Process training data (relabeling PIDs).
        5. Process test data (keeping original PIDs).
        6. Print comprehensive dataset statistics.

        Args:
            dataset_dir (str): Root directory of the dataset.
            min_seq_len (int): Minimum number of frames required for a valid tracklet.
        """

        """ Dataset Catalog Structure
        dataset_dir/
        ├── Train/
        │   ├── 0001/
        │   │   ├── ir/
        │   │   │   ├── D2/
        │   │   │   │   ├── 1.jpg
        │   │   │   │   ├── 6.jpg
        │   │   │   │   └── ...
        │   │   │   ├── D4/
        │   │   │   └── ...
        │   │   └── rgb/
        │   ├── 0002/
        │   ├── 0003/
        │   └── ...
        ├── Test/
        │   └── ...
        └── info/
            ├── track_train_info.txt
            ├── track_test_info.txt
            ├── train_name.txt
            ├── test_name.txt
            └── query_IDX.txt
        """

        self.dataset_dir = dataset_dir

        self.dataset_train_dir = os.path.join(dataset_dir, 'Train')
        self.dataset_test_dir = os.path.join(dataset_dir, 'Test')

        self.train_name_path = os.path.join(dataset_dir, 'info/train_name.txt')  # Eg: 0001M2D3T1F1.jpg
        self.track_train_info_path = os.path.join(dataset_dir, 'info/track_train_info.txt')
        # Eg: 2 1 24 0001 3 (m, start_index, end_index, pid, camid)

        self.test_name_path = os.path.join(dataset_dir, 'info/test_name.txt')  # Eg: 0503M2D3T1F1.jpg
        self.track_test_info_path = os.path.join(dataset_dir, 'info/track_test_info.txt')
        # Eg: 2 1 24 0503 3 (modality_id, start_index, end_index, person_id, camera_id)

        self.query_IDX_path = os.path.join(dataset_dir, 'info/query_IDX.txt')  # Eg: 15  (index of test track)

        self._check_before_run()

        train_names = self._get_names(self.train_name_path)  # Num 232496
        track_train = self._get_tracks(self.track_train_info_path)  # Num (11061, 5)

        test_names = self._get_names(self.test_name_path)  # Num 230763
        track_test = self._get_tracks(self.track_test_info_path)  # Num (10802, 5)

        query_IDX = self._get_query_idx(self.query_IDX_path) - 1  # Num 5159 (-1 for zero starting)
        track_query = track_test[query_IDX, :]  # Num (5159, 5)
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]  # Num 5643
        track_gallery = track_test[gallery_IDX, :]  # Num (5643, 5)
        # By Default: Infrared Query & Visible Gallery - Attention !!!!!

        (train_tracks, train_track_pids, train_track_cids, train_track_mids,
         train_num_imgs_per_track,
         train_num_pids) = self._process_data(train_names, track_train,
                                              relabel=True, min_seq_len=min_seq_len, mode='train')

        num_train_tracks = len(train_tracks)  # Num 9751
        num_train_images = sum(train_num_imgs_per_track)  # Num 224757
        train_ir_slice_mask = np.array(train_track_mids) == 1
        train_rgb_slice_mask = np.array(train_track_mids) == 2
        num_train_ir_tracks = int(sum(train_ir_slice_mask))  # Num 4291
        num_train_ir_images = int(sum(np.array(train_num_imgs_per_track)[train_ir_slice_mask]))  # Num 98214
        num_train_rgb_tracks = int(sum(train_rgb_slice_mask))  # Num 5460
        num_train_rgb_images = int(sum(np.array(train_num_imgs_per_track)[train_rgb_slice_mask]))  # Num 126543

        (query_tracks, query_track_pids, query_track_cids, query_track_mids,
         query_num_imgs_per_track,
         query_num_pids) = self._process_data(test_names, track_query,
                                              relabel=False, min_seq_len=min_seq_len, mode='test')

        (gallery_tracks, gallery_track_pids, gallery_track_cids, gallery_track_mids,
         gallery_num_imgs_per_track,
         gallery_num_pids) = self._process_data(test_names, track_gallery,
                                                relabel=False, min_seq_len=min_seq_len, mode='test')

        assert query_num_pids == gallery_num_pids

        num_query_tracks = len(query_tracks)  # Num 4584
        num_gallery_tracks = len(gallery_tracks)  # Num 5099
        num_query_images = sum(query_num_imgs_per_track)  # Num 105979
        num_gallery_images = sum(gallery_num_imgs_per_track)  # Num 118137

        self.train_tracks = train_tracks  # [(img_path1, img_path2, ...), ...]
        self.train_track_pids = train_track_pids
        self.train_track_cids = train_track_cids
        self.train_track_mids = train_track_mids

        self.query_tracks = query_tracks  # [(img_path1, img_path2, ...), ...]
        self.query_track_pids = query_track_pids
        self.query_track_cids = query_track_cids
        self.query_track_mids = query_track_mids
        self.query_num_pids = query_num_pids

        self.gallery_tracks = gallery_tracks  # [(img_path1, img_path2, ...), ...]
        self.gallery_track_pids = gallery_track_pids
        self.gallery_track_cids = gallery_track_cids
        self.gallery_track_mids = gallery_track_mids
        self.gallery_num_pids = gallery_num_pids

        self.train_num_pids = train_num_pids
        self.test_num_pids = query_num_pids

        print("+-----------------------------------------------+")
        print("|  HITSZ-VCM Dataset                            |")
        print("+-----------------------------------------------+")
        print("| # Subset        | # ids | # tracks | # images |")
        print("+-----------------------------------------------+")
        print("| train           | {:5d} | {:8d} | {:8d} |".format(
            train_num_pids, num_train_tracks, num_train_images))
        print("| - train (ir)    | {:5d} | {:8d} | {:8d} |".format(
            train_num_pids, num_train_ir_tracks, num_train_ir_images))
        print("| - train (rgb)   | {:5d} | {:8d} | {:8d} |".format(
            train_num_pids, num_train_rgb_tracks, num_train_rgb_images))
        print("+-----------------------------------------------+")
        print("| test            | {:5d} | {:8d} | {:8d} |".format(
            query_num_pids, num_query_tracks + num_gallery_tracks, num_query_images + num_gallery_images))
        print("| - query (ir)    | {:5d} | {:8d} | {:8d} |".format(
            query_num_pids, num_query_tracks, num_query_images))
        print("| - gallery (rgb) | {:5d} | {:8d} | {:8d} |".format(
            gallery_num_pids, num_gallery_tracks, num_gallery_images))
        print("+-----------------------------------------------+")

    def _check_before_run(self):
        """
        Checks if all necessary dataset files and directories exist.

        Raises:
            RuntimeError: If any required file or directory is missing.
        """

        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not os.path.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not os.path.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not os.path.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not os.path.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        """
        Reads a text file containing a list of names (e.g., image filenames).

        Args:
            fpath (str): Path to the text file.

        Returns:
            list: A list of strings stripped of newline characters.
        """

        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _get_tracks(self, fpath):
        """
        Reads track metadata from a text file.
        File format expectation: Space-separated integers.

        Args:
            fpath (str): Path to the track info file.

        Returns:
            ndarray: A numpy array where each row represents track metadata.
        """

        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line_items = new_line.split(' ')
                new_line_items = list(map(int, new_line_items))
                names.append(new_line_items)
        names = np.array(names)
        return names

    def _get_query_idx(self, fpath):
        """
        Reads the query indices from a text file.

        Args:
            fpath (str): Path to the query index file.

        Returns:
            ndarray: A numpy array of integer indices.
        """

        idxs = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line_items = new_line.split(' ')
                new_line_items = list(map(int, new_line_items))
                idxs.extend(new_line_items)
        idxs = np.array(idxs)
        return idxs

    def _decoder_pic_path(self, fname):
        """
        Decodes metadata from the filename to construct the relative file path.

        Process:
        1. Locate indices of delimiters 'M', 'D', 'T', and 'F' in the filename.
        2. Extract Person ID (PID), Modality, Camera ID, and Image name by slicing the string.
        3. Map the modality ID ('1' -> 'ir', others -> 'rgb').
        4. Construct the structured file path string.

        Args:
            fname (str): The raw filename string (e.g., '0001M2D3T1F1.jpg').
                         Expected Format: PID + 'M' + ModalityID + 'D' + CameraID + 'T' + TrackID + 'F' + FrameID.

        Returns:
            str: The formatted relative path (e.g., '0001/rgb/D3/1.jpg').
        """

        M_pos = fname.find('M')
        D_pos = fname.find('D')
        T_pos = fname.find('T')
        F_pos = fname.find('F')
        pid = fname[0:M_pos]
        modality = 'ir' if fname[M_pos + 1:D_pos] == '1' else 'rgb'
        camera = fname[D_pos:T_pos]
        image = fname[F_pos + 1:]
        path = f'{pid}/{modality}/{camera}/{image}'
        return path

    def _process_data(self, names, track_meta_data, relabel=False, min_seq_len=0, mode='train'):
        """
        Constructs tracklets from metadata and image names.

        Args:
            names (list): List of image names.
            track_meta_data (ndarray): Array containing track info (mid, start, end, pid, cid).
            relabel (bool): Whether to map PIDs to a continuous range [0, N-1] (for training).
            min_seq_len (int): Filter out tracks shorter than this length.
            mode (str): 'train' or 'test'.

        Returns:
            tuple: (track_paths, pids, cids, mids, num_imgs, num_pids)
        """

        assert mode in ['train', 'test']
        num_tracks = track_meta_data.shape[0]
        pid_list = list(set(track_meta_data[:, 3].tolist()))
        num_pids = len(pid_list)  # Num 500
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}

        track_paths = []
        track_pids = []
        track_cids = []
        track_mids = []
        num_imgs_per_track = []

        for track_idx in range(num_tracks):
            # modality_id, start_index, end_index, person_id, camera_id
            mid, start_index, end_index, pid, cid = track_meta_data[track_idx, ...]
            if relabel: pid = pid2label[pid]

            img_names = names[start_index - 1:end_index]
            img_root_dir = self.dataset_train_dir if mode == 'train' else self.dataset_test_dir
            img_paths = [os.path.join(img_root_dir, self._decoder_pic_path(img_name)) for img_name in img_names]

            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                track_paths.append(img_paths)
                track_pids.append(pid)
                track_cids.append(cid)
                track_mids.append(mid)
                num_imgs_per_track.append(len(img_paths))

        return track_paths, track_pids, track_cids, track_mids, num_imgs_per_track, num_pids


class BUPTCampusDataManager:
    """
    Data Manager for the BUPTCampus dataset.
    It parses the specific folder structure and json files
    to extract identities, cameras, and modalities.
    """

    def __init__(self, dataset_dir):
        """
        Initialize the BUPTCampus data manager.

        Process:
        1. Verify paths and load `data_paths.json`.
        2. Parse Training, Auxiliary, Query, and Gallery lists from text files.
        3. Map string-based Camera IDs (e.g., 'LS3') and Modalities to integers.
        4. Relabel PIDs for the training set.
        5. Print comprehensive dataset statistics.

        Args:
            dataset_dir (str): Root directory of the dataset.
        """

        """ Dataset Catalog Structure
        dataset_dir/
        ├── DATA/
        │   ├── 1/
        │   │   ├── IR/
        │   │   │   ├── CQ1/
        │   │   │   │   ├── 1_IR_CQ1_1_8535.jpg
        │   │   │   │   ├── 1_IR_CQ1_1_8536.jpg
        │   │   │   │   └── ...
        │   │   │   ├── G25/
        │   │   │   └── ...
        │   │   ├── RGB/
        │   │   │   ├── CQ1/
        │   │   │   │   ├── 1_RGB_CQ1_1_8535.jpg
        │   │   │   │   ├── 1_RGB_CQ1_1_8536.jpg
        │   │   │   │   └── ...
        │   │   │   ├── G25/
        │   │   │   └── ...
        │   │   └── FakeIR/
        │   │       ├── CQ1/
        │   │       │   ├── 1_RGB_CQ1_1_8535.jpg
        │   │       │   ├── 1_RGB_CQ1_1_8536.jpg
        │   │       │   └── ...
        │   │       ├── G25/
        │   │       └── ...
        │   ├── 2/
        │   ├── 3/
        │   └── ...
        ├── data_paths.json
        ├── train.txt
        ├── train_aux.txt
        ├── query.txt
        └── gallery.txt
        """

        self.dataset_dir = dataset_dir

        self.dataset_data_dir = os.path.join(dataset_dir, 'DATA')
        self.data_info_path = os.path.join(dataset_dir, 'data_paths.json')
        # Eg: {'1': {'TR': {'CQ1': {'1': ['1/IR/CQ1/1_IR_CQ1_1_8535.jpg', ... ] ... } ... } ... } ... }
        self.train_info_path = os.path.join(dataset_dir, 'train.txt')
        # Eg: 1746 RGB/IR TSG2 1  (pid mid cid tid)
        self.train_aux_info_path = os.path.join(dataset_dir, 'train_auxiliary.txt')
        # Eg: 2830 RGB/IR TSG1 1  (pid mid cid tid)
        self.query_info_path = os.path.join(dataset_dir, 'query.txt')
        # Eg: 1911 RGB G25 1  (pid mid cid tid)
        self.gallery_info_path = os.path.join(dataset_dir, 'gallery.txt')
        # Eg: 1911 IR LS3 1  (pid mid cid tid)

        # Note: Query & Gallery contain samples of both visible and infrared modalities,
        # which need to be filtered when used. - Attention !!!!!

        self._check_before_run()

        data_info = json.load(open(self.data_info_path))

        (train_tracks,
         train_track_pids,
         train_track_cids,
         train_track_mids) = self._parse_data(self.train_info_path, data_info, relabel=True)

        train_num_pids = len(set(train_track_pids))  # Num 1074
        num_train_tracks = len(train_tracks)  # Num 7148
        num_train_images = sum([len(ts) for ts in train_tracks])  # Num 796138
        num_train_ir_tracks = sum([mid == 1 for mid in train_track_mids])  # Num 3574
        num_train_ir_images = sum(
            [len(ts) for i, ts in enumerate(train_tracks) if train_track_mids[i] == 1])  # Num 398069
        num_train_rgb_tracks = sum([mid == 2 for mid in train_track_mids])  # Num 3574
        num_train_rgb_images = sum(
            [len(ts) for i, ts in enumerate(train_tracks) if train_track_mids[i] == 2])  # Num 398069

        (train_aux_tracks,
         train_aux_track_pids,
         train_aux_track_cids,
         train_aux_track_mids) = self._parse_data(self.train_aux_info_path, data_info, relabel=True)

        train_aux_num_pids = len(set(train_aux_track_pids))  # Num 930
        num_train_aux_tracks = len(train_aux_tracks)  # Num 1860
        num_train_aux_images = sum([len(ts) for ts in train_aux_tracks])  # Num 209150
        num_train_aux_ir_tracks = sum([mid == 1 for mid in train_aux_track_mids])  # Num 930
        num_train_aux_ir_images = sum(
            [len(ts) for i, ts in enumerate(train_aux_tracks) if train_aux_track_mids[i] == 1])  # Num 104575
        num_train_aux_rgb_tracks = sum([mid == 2 for mid in train_aux_track_mids])  # Num 930
        num_train_aux_rgb_images = sum(
            [len(ts) for i, ts in enumerate(train_aux_tracks) if train_aux_track_mids[i] == 2])  # Num 104575

        (query_tracks,
         query_track_pids,
         query_track_cids,
         query_track_mids) = self._parse_data(self.query_info_path, data_info, relabel=False)

        query_num_pids = len(set(query_track_pids))  # Num 1076
        num_query_tracks = len(query_tracks)  # Num 1076
        num_query_images = sum([len(ts) for ts in query_tracks])  # Num 101938
        num_query_ir_tracks = sum([mid == 1 for mid in query_track_mids])  # Num 536
        num_query_ir_images = sum(
            [len(ts) for i, ts in enumerate(query_tracks) if query_track_mids[i] == 1])  # Num 53417
        num_query_rgb_tracks = sum([mid == 2 for mid in query_track_mids])  # Num 540
        num_query_rgb_images = sum(
            [len(ts) for i, ts in enumerate(query_tracks) if query_track_mids[i] == 2])  # Num 48521

        (gallery_tracks,
         gallery_track_pids,
         gallery_track_cids,
         gallery_track_mids) = self._parse_data(self.gallery_info_path, data_info, relabel=False)

        gallery_num_pids = len(set(gallery_track_pids))  # Num 1076
        num_gallery_tracks = len(gallery_tracks)  # Num 4844
        num_gallery_images = sum([len(ts) for ts in gallery_tracks])  # Num 597542
        num_gallery_ir_tracks = sum([mid == 1 for mid in gallery_track_mids])  # Num 2422
        num_gallery_ir_images = sum(
            [len(ts) for i, ts in enumerate(gallery_tracks) if gallery_track_mids[i] == 1])  # Num 298771
        num_gallery_rgb_tracks = sum([mid == 2 for mid in gallery_track_mids])  # Num 2422
        num_gallery_rgb_images = sum(
            [len(ts) for i, ts in enumerate(gallery_tracks) if gallery_track_mids[i] == 2])  # Num 298771

        assert query_num_pids == gallery_num_pids

        self.train_tracks = train_tracks  # [(img_path1, img_path2, ...), ...]
        self.train_track_pids = train_track_pids
        self.train_track_cids = train_track_cids
        self.train_track_mids = train_track_mids

        ## Use the auxiliary training set [Optional - Not used in M3-ReID]
        # self.train_tracks = train_tracks + train_aux_tracks  # [(img_path1, img_path2, ...), ...]
        # self.train_track_pids = train_track_pids + [p + len(train_track_pids) for p in train_aux_track_pids]
        # self.train_track_cids = train_track_cids + train_aux_track_cids
        # self.train_track_mids = train_track_mids + train_aux_track_mids

        self.query_tracks = query_tracks  # [(img_path1, img_path2, ...), ...]
        self.query_track_pids = query_track_pids
        self.query_track_cids = query_track_cids
        self.query_track_mids = query_track_mids
        self.query_num_pids = query_num_pids

        self.gallery_tracks = gallery_tracks  # [(img_path1, img_path2, ...), ...]
        self.gallery_track_pids = gallery_track_pids
        self.gallery_track_cids = gallery_track_cids
        self.gallery_track_mids = gallery_track_mids
        self.gallery_num_pids = gallery_num_pids

        self.train_num_pids = train_num_pids
        ## Use the auxiliary training set [Optional - Not used in M3-ReID]
        # self.train_num_pids = train_num_pids + train_aux_num_pids
        self.test_num_pids = query_num_pids

        print("+---------------------------------------------------+")
        print("|  BUPTCampus Dataset                               |")
        print("+---------------------------------------------------+")
        print("| # Subset            | # ids | # tracks | # images |")
        print("+---------------------------------------------------+")
        print("| train               | {:5d} | {:8d} | {:8d} |".format(
            train_num_pids, num_train_tracks, num_train_images))
        print("| - train (ir)        | {:5d} | {:8d} | {:8d} |".format(
            train_num_pids, num_train_ir_tracks, num_train_ir_images))
        print("| - train (rgb)       | {:5d} | {:8d} | {:8d} |".format(
            train_num_pids, num_train_rgb_tracks, num_train_rgb_images))
        print("+---------------------------------------------------+")
        print("| train_aux           | {:5d} | {:8d} | {:8d} |".format(
            train_aux_num_pids, num_train_aux_tracks, num_train_aux_images))
        print("| - train_aux (ir)    | {:5d} | {:8d} | {:8d} |".format(
            train_aux_num_pids, num_train_aux_ir_tracks, num_train_aux_ir_images))
        print("| - train_aux (rgb)   | {:5d} | {:8d} | {:8d} |".format(
            train_aux_num_pids, num_train_aux_rgb_tracks, num_train_aux_rgb_images))
        print("+---------------------------------------------------+")
        print("| test_query          | {:5d} | {:8d} | {:8d} |".format(
            query_num_pids, num_query_tracks, num_query_images))
        print("| - test_query (ir)   | {:5d} | {:8d} | {:8d} |".format(
            query_num_pids, num_query_ir_tracks, num_query_ir_images))
        print("| - test_query (rgb)  | {:5d} | {:8d} | {:8d} |".format(
            query_num_pids, num_query_rgb_tracks, num_query_rgb_images))
        print("+---------------------------------------------------+")
        print("| test_gallery        | {:5d} | {:8d} | {:8d} |".format(
            gallery_num_pids, num_gallery_tracks, num_gallery_images))
        print("| - test_gallery (ir) | {:5d} | {:8d} | {:8d} |".format(
            gallery_num_pids, num_gallery_ir_tracks, num_gallery_ir_images))
        print("| - test_gallery (rgb)| {:5d} | {:8d} | {:8d} |".format(
            gallery_num_pids, num_gallery_rgb_tracks, num_gallery_rgb_images))
        print("+---------------------------------------------------+")

    def _check_before_run(self):
        """
        Checks if all necessary dataset files and directories exist.

        Raises:
            RuntimeError: If any required file or directory is missing.
        """

        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.dataset_data_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_data_dir))
        if not os.path.exists(self.data_info_path):
            raise RuntimeError("'{}' is not available".format(self.data_info_path))
        if not os.path.exists(self.train_info_path):
            raise RuntimeError("'{}' is not available".format(self.train_info_path))
        if not os.path.exists(self.train_aux_info_path):
            raise RuntimeError("'{}' is not available".format(self.train_aux_info_path))
        if not os.path.exists(self.query_info_path):
            raise RuntimeError("'{}' is not available".format(self.query_info_path))
        if not os.path.exists(self.gallery_info_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_info_path))

    def _parse_data(self, path, data_info, relabel=False):
        """
        Parses the dataset info file and retrieves image paths from the loaded JSON data.

        Args:
            path (str): Path to the text file containing the list (pid mid cid tid).
            data_info (dict): The loaded JSON object containing file paths.
            relabel (bool): Whether to map PIDs to a continuous range.

        Returns:
            tuple: (tracks, pids, cids, mids) where IDs are converted to integers.
        """

        tracks, pids, cids, mids = [], [], [], []
        with open(path) as f:
            for line in f:
                pid, mid, cid, tid = line.strip().split(' ')
                if mid == 'RGB/IR':
                    tracks.append(data_info[pid]['IR'][cid][tid])
                    pids.append(pid)
                    cids.append(cid)
                    mids.append('IR')
                    tracks.append(data_info[pid]['RGB'][cid][tid])
                    pids.append(pid)
                    cids.append(cid)
                    mids.append('RGB')
                else:
                    tracks.append(data_info[pid][mid][cid][tid])
                    pids.append(pid)
                    cids.append(cid)
                    mids.append(mid)
        tracks = [[os.path.join(self.dataset_data_dir, t) for t in ts] for ts in tracks]
        if relabel: pid2label = {pid: label for label, pid in enumerate(set(pids))}
        pids = [pid2label[p] if relabel else int(p) for p in pids]
        cam2cid = {'LS3': 0, 'G25': 1, 'CQ1': 2, 'W4': 3, 'TSG1': 4, 'TSG2': 5}
        cids = [cam2cid[c] for c in cids]
        mod2mid = {'IR': 1, 'RGB': 2}
        mids = [mod2mid[m] for m in mids]
        return tracks, pids, cids, mids
