__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

from . import efficient

EfficientSeg = efficient.EfficientSeg
EFFLoss = efficient.EffLoss


def build_model(cfg):
    if cfg.model_type == "eff":
        model = EfficientSeg(cfg.data.num_classes, compound_coef=cfg.compound_coef,
                     ratios=eval(cfg.anchors_ratios), scales=eval(cfg.anchors_scales))
    else:
        raise RuntimeError("no support for model type:%s"%cfg.model_type)
    return model


def build_loss(cfg, device):
    if cfg.model_type == "eff":
        loss = EFFLoss(device)
    else:
        raise RuntimeError("no support for model type:%s"%cfg.model_type)
    return loss
