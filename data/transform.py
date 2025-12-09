# ------------------------------------------------------------------------------
# File:    M3-ReID/data/transform.py
#
# Description:
#    This module implements custom data transformation and augmentation techniques
#    specifically designed for visible-infrared ReID.
#
# Key Features:
# - Synchronized transformations across all frames in a video track.
# - Weighted Grayscale conversion for modality augmentation.
# - Style Variation to simulate diverse lighting conditions.
#
# Classes:
# - SyncTrackTransform
# - WeightedGrayscale
# - StyleVariation
# ------------------------------------------------------------------------------

import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from tools.utils import set_seed


class SyncTrackTransform(nn.Module):
    """
    Wraps a standard transform to apply it consistently across a sequence of frames (video track).

    Note that some previous VVI-ReID methods apply random augmentations independently per frame, which
    may inadvertently disrupt temporal continuity. We fix this potential bug and ensure that operations
    (e.g., random crop, flip) are identical for all frames in a track to preserve temporal integrity.
    """

    def __init__(self, transform):
        """
        Initialize the Synchronized Track Transform.

        Args:
            transform (callable): The transformation pipeline to apply to each frame.
        """

        super().__init__()
        self.transform = transform

    def forward(self, track):
        """
        Applies the transform to the entire video track.

        Process:
        1. Generate a random seed.
        2. Iterate through each image in the track.
        3. Reset the random seed before processing each image to ensure identical transformation parameters.
        4. Apply the transform and stack the results into a single tensor.

        Args:
            track (list): A list of image data (e.g., PIL Images or Arrays).

        Returns:
            Tensor: A stacked tensor representing the transformed video track [T, C, H, W].
        """

        seed = random.randint(0, 2 ** 32 - 1)

        track_data = []
        for img_data in track:
            set_seed(seed)
            img_data = np.array(img_data)
            img_data = self.transform(img_data)
            track_data.append(img_data)
        track_data = torch.stack(track_data, dim=0)

        return track_data


class WeightedGrayscale(nn.Module):
    """
    Converts an RGB image to grayscale using custom or randomized channel weights.
    This augmentation bridges the modality gap by simulating infrared-like single-channel characteristics.

    'Bridging the Gap: Multi-Level Cross-Modality Joint Alignment for Visible-Infrared Person Re-Identification'
    by Liang et al. See https://ieeexplore.ieee.org/abstract/document/10472470 (IEEE TCSVT 2024).
    """

    def __init__(self, weights=None, p=1.0):
        """
        Initialize the Weighted Grayscale transform.

        Args:
            weights (tuple or list, optional): Fixed weights for R, G, B channels. If None, weights are randomized.
            p (float): Probability of applying this transform.
        """

        super().__init__()
        self.weights = weights
        self.p = p

    def forward(self, img):
        """
        Applies weighted grayscale conversion to the input image.

        Process:
        1. Check if the transform should be applied based on probability `p`.
        2. If `weights` are not provided, generate random uniform weights for R, G, B.
        3. Normalize weights so they sum to 1.
        4. Compute the weighted sum of channels: w1*R + w2*G + w3*B.
        5. Expand the result back to 3 channels for compatibility.

        Args:
            img (PIL.Image or np.ndarray): Input image.

        Returns:
            PIL.Image: Grayscale converted image (still in 3-channel format).
        """

        if self.p < torch.rand(1):
            return img

        if self.weights is not None:
            w1, w2, w3 = self.weights
        else:
            w1 = random.uniform(0, 1)
            w2 = random.uniform(0, 1)
            w3 = random.uniform(0, 1)
            s = w1 + w2 + w3
            w1, w2, w3 = w1 / s, w2 / s, w3 / s
        img_data = np.asarray(img)
        img_data = w1 * img_data[:, :, 0] + w2 * img_data[:, :, 1] + w3 * img_data[:, :, 2]
        img_data = np.expand_dims(img_data, axis=-1).repeat(3, axis=-1)

        return Image.fromarray(np.uint8(img_data))


class StyleVariation(nn.Module):
    """
    Randomly varies the intensity of image channels to simulate style or lighting changes.

    'Video-based Visible-Infrared Person Re-Identification via Style Disturbance Defense and Dual Interaction'
    by Zhou et al. See https://dl.acm.org/doi/abs/10.1145/3581783.3612479 (ACM MM 2023).
    """

    def __init__(self, mode='all', p=1.0):
        """
        Initialize the Style Variation transform.

        Args:
            mode (str): 'all' to vary channels independently, 'one' to vary them uniformly.
            p (float): Probability of applying this transform.
        """

        super().__init__()
        assert mode in ['all', 'one']
        self.mode = mode
        self.p = p

    def forward(self, img):
        """
        Applies style variation to the input tensor.

        Process:
        1. Check if the transform should be applied based on probability `p`.
        2. Generate random scaling weights (uniform distribution between 0.5 and 1.5).
        3. Reshape weights based on the `mode` (per-channel or global).
        4. Multiply the input image tensor by these weights.

        Args:
            img (Tensor): Input image tensor.

        Returns:
            Tensor: Stylized image tensor.
        """

        if self.p < torch.rand(1):
            return img

        if self.mode == 'all':
            style_variation_weights = torch.FloatTensor(3).uniform_(0.5, 1.5)
            style_variation_weights = style_variation_weights.view(3, 1, 1)
        else:  # self.mode == 'one'
            style_variation_weights = torch.FloatTensor(1).uniform_(0.5, 1.5)

        return style_variation_weights * img
