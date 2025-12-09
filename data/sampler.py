# ------------------------------------------------------------------------------
# File:    M3-ReID/data/sampler.py
#
# Description:
#    This module implements various custom Samplers for PyTorch DataLoaders.
#    These samplers control how batches are formed, ensuring specific constraints
#    like PxK sampling (P identities, K instances) and balancing modalities
#    (Visible/Infrared) within batches.
#
# Classes:
# - NormTripletSampler: Standard PxK sampling without modality constraints.
# - CrossModalityTripletSampler: PxK sampling with modality ratio control.
# - CrossModalityRandomSampler: Half IR, Half RGB random sampling.
# - CrossModalityIdentitySampler: Structured PxK sampling separating modalities in halves.
# - IdentityCrossModalitySampler: PxK sampling interleaving modalities per ID.
# ------------------------------------------------------------------------------

import copy
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class NormTripletSampler(Sampler):
    """
    Randomly sample P identities, then for each identity,
    randomly sample K instances, therefore batch size is P*K.
    It does not distinguish modalities.
    """

    def __init__(self, dataset, batch_size, num_instances):
        """
        Initialize the Standard Triplet Sampler.

        Args:
            dataset (Dataset): The dataset object containing PIDs and metadata.
            batch_size (int): The total size of the batch (P*K).
            num_instances (int): The number of instances per identity (K).
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.dataset.pids):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        """
        Generates an iterator for epoch-level sampling.

        Process:
        1. For each PID, select K instances (with replacement if needed).
        2. Construct batches by selecting PIDs until all distinct PIDs are exhausted.

        Returns:
            iter: An iterator yielding indices for the DataLoader.
        """

        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        """
        Returns the estimated number of samples in an epoch.

        Returns:
            int: Total length of samples.
        """

        return self.length


class CrossModalityTripletSampler(Sampler):
    """
    Randomly sample P identities, then for each identity,
    randomly sample K instances, controlling the modal ratio for modalities.
    """

    def __init__(self, dataset, batch_size, num_instances, modal_ratio=0.5):
        """
        Initialize the Cross-Modality Triplet Sampler.

        Args:
            dataset (Dataset): The dataset object containing PIDs and modality IDs.
            batch_size (int): The total size of the batch.
            num_instances (int): The number of instances per identity.
            modal_ratio (float, optional): The target ratio of Infrared samples (default is 0.5).
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.modal_ratio = modal_ratio
        self.index_dic = defaultdict(list)
        for index, (pid, mid) in enumerate(zip(self.dataset.pids, self.dataset.mids)):
            self.index_dic[pid].append((index, mid))
        self.pids = list(self.index_dic.keys())

        # Estimate number of examples in an epoch
        self.length = 0
        self.__iter__()

    def __iter__(self):
        """
        Generates an iterator ensuring modality balance within each identity's samples.

        Process:
        1. Separate indices into IR and RGB lists for each PID.
        2. Sample `num_instances * modal_ratio` IR images and remainder RGB images.
        3. Shuffle and combine to form the batch.

        Returns:
            iter: An iterator yielding indices.
        """

        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])

            ir_idxs = [idx[0] for idx in idxs if idx[1] == 1]
            rgb_idxs = [idx[0] for idx in idxs if idx[1] == 2]
            num_ir_samples = int(self.num_instances * self.modal_ratio)
            num_rgb_samples = self.num_instances - num_ir_samples

            if len(ir_idxs) < num_ir_samples:
                ir_idxs = np.random.choice(ir_idxs, size=num_ir_samples, replace=True)
            if len(rgb_idxs) < num_rgb_samples:
                rgb_idxs = np.random.choice(rgb_idxs, size=num_rgb_samples, replace=True)

            num_repeats = (len(ir_idxs) + len(rgb_idxs)) // self.num_instances

            for _ in range(num_repeats):
                batch_ir_idxs = np.random.choice(ir_idxs, size=num_ir_samples, replace=True)
                batch_rgb_idxs = np.random.choice(rgb_idxs, size=num_rgb_samples, replace=True)
                batch_idxs = np.hstack((batch_ir_idxs, batch_rgb_idxs))
                np.random.shuffle(batch_idxs)
                batch_idxs_dict[pid].append(batch_idxs)

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        """
        Returns the estimated number of samples in an epoch.

        Returns:
            int: Total length of samples.
        """

        return self.length


class CrossModalityRandomSampler(Sampler):
    """
    The first half of a batch are randomly selected IR images,
    and the second half are randomly selected RGB images.
    """

    def __init__(self, dataset, batch_size):
        """
        Initialize the Cross-Modality Random Sampler.

        Args:
            dataset (Dataset): The dataset object.
            batch_size (int): The size of the batch.
        """

        self.dataset = dataset
        self.batch_size = batch_size

        self.ir_list = [i for i, m in enumerate(dataset.mids) if m == 1]
        self.rgb_list = [i for i, m in enumerate(dataset.mids) if m == 2]

    def __iter__(self):
        """
        Generates an iterator with a fixed structure: [Half IR Batch, Half RGB Batch].

        Process:
        1. Permute IR and RGB indices separately.
        2. Pad the smaller modality to match the larger one.
        3. Construct the batch by concatenating a chunk of IR with a chunk of RGB.

        Returns:
            iter: An iterator yielding indices.
        """

        sample_list = []
        ir_list = np.random.permutation(self.ir_list).tolist()
        rgb_list = np.random.permutation(self.rgb_list).tolist()

        rgb_size = len(self.rgb_list)
        ir_size = len(self.ir_list)
        if rgb_size >= ir_size:
            diff = rgb_size - ir_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                ir_list.extend(np.random.permutation(self.ir_list).tolist())
            ir_list.extend(np.random.choice(self.ir_list, pad_size, replace=False).tolist())
        else:
            diff = ir_size - rgb_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                rgb_list.extend(np.random.permutation(self.rgb_list).tolist())
            rgb_list.extend(np.random.choice(self.rgb_list, pad_size, replace=False).tolist())

        assert len(ir_list) == len(rgb_list)

        half_bs = self.batch_size // 2
        for start in range(0, len(ir_list), half_bs):
            sample_list.extend(ir_list[start:start + half_bs])
            sample_list.extend(rgb_list[start:start + half_bs])

        return iter(sample_list)

    def __len__(self):
        """
        Returns the estimated number of samples in an epoch.

        Returns:
            int: Total length of samples.
        """

        return max(len(self.rgb_list), len(self.ir_list)) * 2


class CrossModalityIdentitySampler(Sampler):
    """
    The first half of a batch are randomly selected k_size/2 IR images for each randomly selected p_size people,
    and the second half are randomly selected k_size/2 RGB images for each the same p_size people.
    Batch - [id1_ir, id1_ir, ..., id2_ir, id2_ir, ..., id1_rgb, id1_rgb, ..., id2_rgb, id2_rgb, ...]
    """

    def __init__(self, dataset, p_size, k_size):
        """
        Initialize the Identity-based Cross-Modality Sampler.

        Args:
            dataset (Dataset): The dataset object.
            p_size (int): Number of identities (P) per batch.
            k_size (int): Number of instances (K) per identity (split between IR/RGB).
        """

        self.dataset = dataset
        self.p_size = p_size
        self.k_size = k_size // 2
        self.batch_size = p_size * k_size * 2

        self.id2idx_rgb = defaultdict(list)
        self.id2idx_ir = defaultdict(list)
        for i, identity in enumerate(dataset.pids):
            if dataset.mids[i] == 1:
                self.id2idx_ir[identity].append(i)
            else:  # dataset.mids[i] == 2
                self.id2idx_rgb[identity].append(i)

        self.num_base_samples = self.dataset.num_unique_pids * self.k_size * 2

        self.num_repeats = len(dataset.pids) // self.num_base_samples
        self.num_samples = self.num_base_samples * self.num_repeats

    def __iter__(self):
        """
        Generates an iterator structured by identity and split by modality blocks.

        Process:
        1. Select `p_size` identities.
        2. For each identity, sample `k_size/2` IR images.
        3. For the same identities, sample `k_size/2` RGB images.
        4. Concatenate the IR block followed by the RGB block.

        Returns:
            iter: An iterator yielding indices.
        """

        sample_list = []

        for r in range(self.num_repeats):
            id_perm = np.random.permutation(self.dataset.num_unique_pids)
            for start in range(0, self.dataset.num_unique_pids, self.p_size):
                selected_ids = id_perm[start:start + self.p_size]

                sample = []
                for identity in selected_ids:
                    replace = len(self.id2idx_ir[identity]) < self.k_size
                    s = np.random.choice(self.id2idx_ir[identity], size=self.k_size, replace=replace)
                    sample.extend(s)
                sample_list.extend(sample)

                sample.clear()
                for identity in selected_ids:
                    replace = len(self.id2idx_rgb[identity]) < self.k_size
                    s = np.random.choice(self.id2idx_rgb[identity], size=self.k_size, replace=replace)
                    sample.extend(s)
                sample_list.extend(sample)

        return iter(sample_list)

    def __len__(self):
        """
        Returns the estimated number of samples in an epoch.

        Returns:
            int: Total length of samples.
        """

        return self.num_samples


class IdentityCrossModalitySampler(Sampler):
    """
    It is equivalent to CrossModalityIdentitySampler, but the arrangement is different.
    Batch - [id1_ir, id1_rgb, id1_ir, id1_rgb, ..., id2_ir, id2_rgb, id2_ir, id2_rgb, ...]
    """

    def __init__(self, dataset, batch_size, num_instances):
        """
        Initialize the Interleaved Identity Sampler.

        Args:
            dataset (Dataset): The dataset object.
            batch_size (int): The total batch size.
            num_instances (int): Number of instances per identity.
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_R = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        for i, identity in enumerate(dataset.pids):
            if dataset.mids[i] == 1:
                self.index_dic_I[identity].append(i)
            else:  # dataset.mids[i] == 2
                self.index_dic_R[identity].append(i)
        self.unique_pids = list(self.index_dic_I.keys())

        # estimate number of examples in an epoch
        self.length = 0
        self.__iter__()

    def __iter__(self):
        """
        Generates an iterator interleaving IR and RGB samples for each identity.

        Process:
        1. For each identity, sample equal numbers of IR and RGB images.
        2. Shuffle them internally but keep them paired in the final list.
        3. Ensure the batch contains `num_instances` samples per PID.

        Returns:
            iter: An iterator yielding indices.
        """

        batch_idxs_dict = defaultdict(list)

        for pid in self.unique_pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_R = copy.deepcopy(self.index_dic_R[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_R) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_R = np.random.choice(idxs_R, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_R):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_R), replace=False)
            if len(idxs_R) > len(idxs_I):
                idxs_R = np.random.choice(idxs_R, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_R)
            batch_idxs = []
            for idx_I, idx_R in zip(idxs_I, idxs_R):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_R)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = [key for key in batch_idxs_dict if len(batch_idxs_dict[key]) > 0]
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        """
        Returns the estimated number of samples in an epoch.

        Returns:
            int: Total length of samples.
        """

        return self.length
