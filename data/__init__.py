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
from data.dataset import is_train_phase, is_val_phase
from data import cityscapes, coco
from data.dir import DirDatasetBuilder


datasetBuildersMap = {
    "cityscapes": cityscapes.CityscapesDatasetBuilder,
    "coco": coco.COCODatasetBuilder,
    "dir": DirDatasetBuilder
}

datasetClsNumMap = {
    "cityscapes": cityscapes.num_cls,
    "coco": coco.num_cls
}


def get_cls_num(datatype):
    return datasetClsNumMap[datatype]


def collate_fn_with_label(batch):
    """
    merge the batch inputs
    :param batch:
    :return:
    """
    batch_inputs = [e for e in zip(*batch)]
    input_tensors = torch.stack(batch_inputs[0])
    labels = [e for e in zip(*batch_inputs[1])]
    trans_infos = batch_inputs[2]
    return input_tensors, labels, trans_infos


def collate_fn_without_label(batch):
    """
    merge the batch inputs
    :param batch:
    :return:
    """
    batch_inputs = [e for e in zip(*batch)]
    input_tensors = torch.stack(batch_inputs[0])
    trans_infos = batch_inputs[1]
    return input_tensors, trans_infos


def get_dataloader(batch_size, dataset_type, data_dir, phase, transforms=None, num_workers=0, random=True, with_label=True):
    """
    initialize the data loader, and then return a data loader
    :param num_workers: worker num
    :param phase: "train", "test", "val"
    :param batch_size:
    :param dataset_type:
    :param data_dir:
    :param random:
    :param transforms:
    :return:
    """
    # determine the class of DatasetBuilder by dataset type
    dataset_builder_class = datasetBuildersMap[dataset_type]
    # initialize dataset
    dataset_builder = dataset_builder_class(data_dir, phase)
    dataset = dataset_builder.get_dataset(transforms=transforms)
    if with_label:
        if is_train_phase(phase):
            # initialize sampler
            if random:
                return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn_with_label
                                                   , num_workers=num_workers)
            else:
                return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_with_label
                                               , num_workers=num_workers)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_with_label
                                                   , num_workers=num_workers)
    else:
        collate_fn = collate_fn_without_label
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           collate_fn=collate_fn
                                           , num_workers=num_workers)
