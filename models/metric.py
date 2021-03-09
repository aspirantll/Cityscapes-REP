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


class MetricCalculator(object):
    """
    compute the metric such as mAP, and show the plot
    """
    def __init__(self, cls_names, iou_th=0.5):
        self._cls_names = cls_names
        self._iou_th = iou_th

    def process(self, outputs, targets):
        pass


