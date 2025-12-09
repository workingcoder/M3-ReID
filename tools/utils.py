# ------------------------------------------------------------------------------
# File:    M3-ReID/tools/utils.py
#
# Description:
#    This module provides general utility functions for the training and evaluation
#    pipeline, including random seeding, I/O handling, and logging.
#
# Key Features:
# - Global random seed setting for reproducibility.
# - Robust image reading with error handling.
# - Directory creation utilities.
# - Logger class for simultaneous console and file output.
#
# Classes:
# - Logger
#
# Main Functions:
# - set_seed
# - read_image
# - mkdir_if_missing
# ------------------------------------------------------------------------------

import os
import sys
import random
import datetime
import errno
import numpy as np
import torch
from PIL import Image


def set_seed(seed, cuda=True):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
        cuda (bool): If True, sets seeds for CUDA backends and forces deterministic algorithms.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def time_str(fmt=None):
    """
    Returns the current timestamp as a formatted string.

    Args:
        fmt (str, optional): The time format string (default '%Y-%m-%d_%H-%M-%S').

    Returns:
        str: Formatted time string.
    """

    if fmt is None:
        fmt = '%Y-%m-%d_%H-%M-%S'
    return datetime.datetime.today().strftime(fmt)


def read_image(img_path):
    """
    Reads an image from the specified path, converting it to RGB.

    Process:
    1. Attempt to open the image using PIL.
    2. If an IOError occurs (common in heavy I/O), catch the exception, print a warning,
       and retry the operation immediately until success.
    3. Convert the loaded image to RGB format.

    Args:
        img_path (str): Path to the image file.

    Returns:
        PIL.Image: The loaded RGB image object.
    """

    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def mkdir_if_missing(dir):
    """
    Creates a directory if it does not already exist.

    Args:
        dir (str): Path to the directory to create.

    Raises:
        OSError: If directory creation fails for reasons other than already existing.
    """

    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger:
    """
    Writes console output to an external text file simultaneously.
    Acts as a wrapper around `sys.stdout`.
    """

    def __init__(self, fpath=None):
        """
        Initialize the Logger.

        Args:
            fpath (str, optional): Path to the log file. If None, only writes to console.
        """

        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def write(self, msg):
        """
        Writes a message to both the console and the log file.

        Args:
            msg (str): The message string to write.
        """

        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        """
        Flushes both the console and file output streams to ensure data is written immediately.
        """

        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        """
        Closes the console and the log file handler.
        """

        self.console.close()
        if self.file is not None:
            self.file.close()
