__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import torch
import numpy as np


def save_labels(input_tensor, label, path):
    centers, cls_ids, polygons, box_sizes, kp_target = label
    kp_arr = kp_target.numpy()
    input_arr = input_tensor.numpy()
    np.savez_compressed(path, (input_arr, centers, cls_ids, polygons, box_sizes, kp_arr))


def load_labels(path):
    input_arr, centers, cls_ids, polygons, box_sizes, kp_arr = np.load(path, allow_pickle=True)['arr_0.npy']
    kp_target = torch.from_numpy(kp_arr)
    input_tensor = torch.from_numpy(input_arr)
    return input_tensor, (centers, cls_ids, polygons, box_sizes, kp_target)
