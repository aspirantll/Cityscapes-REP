
__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import os
import numpy as np
from pycocotools import mask
from torch.utils import data
from utils.image import poly_to_mask
from .dataset import DatasetBuilder
from utils.image import load_rgb_image


skip_cls_ids = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
num_cls = 80


def convert_cls_id_to_index(cls_id):
    """
    convert the class id to cls id
    :param cls_id:
    :return:
    """
    skip_pos = 0
    while skip_pos < len(skip_cls_ids) and skip_cls_ids[skip_pos] < cls_id:
        skip_pos += 1
    return cls_id - skip_pos - 1


def parse_segmentation(ann, img_size):
    """
    parse segmentation
    :param ann: annotation
    :return: list[numpy]
    """
    segm = ann["segmentation"]
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        segment_poly = segm[0]
        segments = np.array(segment_poly, dtype=np.float32).reshape((-1, 2))
        instance_mask = poly_to_mask(segments, img_size)
    else:
        segments = None
        if type(segm['counts']) == list:
            # uncompressed RLE
            rle = mask.frPyObjects(segm, segm["size"][0], segm["size"][1])
        else:
            # rle
            rle = ann['segmentation']
        instance_mask = mask.decode(rle)
    return instance_mask


class COCODataset(data.Dataset):
    """
    the dataset for coco. it includes the augmentation of data.
    """
    def __init__(self, root, transforms=None, subset='train'):
        # save the parameters
        self._data_dir = root
        self._phase = subset
        if transforms is not None:
            self._transforms = transforms  # ADDED THIS
        # initialize the coco api
        from pycocotools.coco import COCO

        ann_file = os.path.join(root, "annotations/instances_%s2017.json"%subset)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # locating index
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = os.path.join(self._data_dir, "%s2017" % self._phase, coco.loadImgs(img_id)[0]['file_name'])
        input_img = load_rgb_image(path)

        height, width, _ = input_img.shape
        class_map = np.zeros((height, width), dtype=np.uint8)
        instance_map = np.zeros((height, width), dtype=np.uint8)
        instance_id = 1
        for ann in anns:
            # handle boundary, reverse point(w,h) to (h,w)
            mask = parse_segmentation(ann, (height, width))
            instance_map = instance_map*(1-mask)+mask*instance_id
            class_map = class_map*(1-mask)+mask*convert_cls_id_to_index(ann["category_id"])

        label = (class_map, instance_map)
        input_img, label, trans_info = self._transforms(input_img, label, path)
        return input_img, label, trans_info

    def __len__(self):
        return len(self.ids)


class COCODatasetBuilder(DatasetBuilder):
    def __init__(self, data_dir, phase="train"):
        super().__init__(data_dir, phase)

    def get_dataset(self, **kwargs):
        return COCODataset(self._data_dir, subset=self._phase, **kwargs)