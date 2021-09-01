
__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import json
import os
import numpy as np
from torch.utils.data import Dataset

import cv2
from utils.tranform import TransInfo
from .dataset import DatasetBuilder
from utils.image import load_rgb_image


label_names = ['background', 'Motor Vehicle', 'Non-motorized Vehicle', 'Pedestrian', 'Traffic Light-Red Light', 'Traffic Light-Yellow Light', 'Traffic Light-Green Light', 'Traffic Light-Off', 'Solid lane line', 'Dotted lane line', 'Crosswalk']
label_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mask_ids = [8, 9, 10]

num_cls = len(label_names)
IMAGE_EXTENSIONS = ['.jpg', '.png']
label_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def is_image(filename):
    return any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_label(filename):
    return filename.endswith('.json')


def load_json_label(label_file, img_size):
    with open(label_file, 'r') as f:
        label_json = json.load(f)

    instance_map = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    boxes_ann = np.empty((0, 5), dtype=np.float32)

    obj_id = 1
    for obj in label_json:
        polygon = np.array(obj['segmentation'], dtype=np.int32).reshape(-1, 2)
        if len(polygon) != 0:
            lt = np.min(polygon, axis=0)
            rb = np.max(polygon, axis=0)
        else:
            lt = np.array((obj['x'], obj['y']), dtype=np.float32)
            rb = lt + np.array((obj['width'], obj['height']), dtype=np.float32)

        wh = rb - lt
        if wh[0]*wh[1] < 4:
            continue
        label_count[obj['type']] = label_count[obj['type']] + 1
        if len(polygon) != 0:
            cv2.fillPoly(instance_map, [polygon], obj_id)
        obj_id = obj_id + 1
        boxes = np.array([[lt[0], lt[1], rb[0], rb[1], obj['type']]], dtype=np.float32)
        boxes_ann = np.append(boxes_ann, boxes, axis=0)
    return boxes_ann, instance_map


class CustomizeDataset(Dataset):

    def __init__(self, root, transforms=None, subset='train'):
        self.images_root = os.path.join(root, subset + '/image')
        self.labels_root = os.path.join(root, subset + '/label')

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f
                          in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in
                            fn if is_label(f)]
        self.filenamesGt.sort()

        self.filenames = self.filenames[:7000]
        self.filenamesGt = self.filenamesGt[:7000]

        self._transforms = transforms  # ADDED THIS

        print("dataset size: {}".format(len(self.filenames)))

    def __getitem__(self, index):
        filename = self.filenames[index]

        img_path = filename
        input_img = load_rgb_image(img_path)
        img_size = input_img.shape[1::-1]

        filenameGt = self.filenamesGt[index]
        label = load_json_label(filenameGt, img_size)

        if self._transforms is not None:
            return self._transforms(input_img, label, img_path=img_path, img_size=img_size)

        return input_img, label, TransInfo(img_path, img_size, 1, img_size, (0, 0))

    def __len__(self):
        return len(self.filenames)


class CustomizeDatasetBuilder(DatasetBuilder):
    def __init__(self, data_dir, phase):
        super().__init__(data_dir, phase)

    def get_dataset(self, **kwargs):
        return CustomizeDataset(self._data_dir, subset=self._phase, **kwargs)
