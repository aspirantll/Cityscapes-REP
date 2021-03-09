__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

from . import fcos, efficient, loss

EfficientSeg = efficient.EfficientSeg
FCOSSeg = fcos.FCOSSeg
EFFLoss = loss.EffLoss
FCOSLoss = loss.FCOSLoss


