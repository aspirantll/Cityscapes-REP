import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset
from collections import namedtuple

from .dataset import DatasetBuilder
from skimage.segmentation import relabel_sequential
from utils.image import load_rgb_image, load_instance_image

label_names = ['background', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
label_ids = [-10, 24, 25, 26, 27, 28, 31, 32, 33]

num_cls = len(label_names)
IMAGE_EXTENSIONS = ['.jpg', '.png']


def is_image(filename):
    return any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_label(filename):
    return "gtFine_instanceIds" in filename


def decode_instance(pic):
    instance_map = np.zeros(
        (pic.shape[0], pic.shape[1]), dtype=np.uint8)

    # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
    class_map = np.zeros(
        (pic.shape[0], pic.shape[1]), dtype=np.uint8)

    for i, c in enumerate(label_ids):
        mask = np.logical_and(pic >= c * 1000, pic < (c + 1) * 1000)
        if mask.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids + np.amax(instance_map)
            class_map[mask] = i+1
    return class_map, instance_map


class CityscapesDataset(Dataset):

    def __init__(self, root, transforms=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)
        self.labels_root = os.path.join(root, 'gtFine/' + subset)

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f
                          in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in
                            fn if is_label(f)]
        self.filenamesGt.sort()

        if transforms is not None:
            self._transforms = transforms  # ADDED THIS

        print("dataset size: {}".format(len(self.filenames)))

    def __getitem__(self, index):
        filename = self.filenames[index]

        img_path = filename
        input_img = load_rgb_image(img_path)

        filenameGt = self.filenamesGt[index]
        instance_img = load_instance_image(filenameGt)
        label = decode_instance(instance_img)

        input_img, label, trans_info = self._transforms(input_img, label, img_path)
        return input_img, label, trans_info

    def __len__(self):
        return len(self.filenames)


class CityscapesDatasetBuilder(DatasetBuilder):
    def __init__(self, data_dir, phase):
        super().__init__(data_dir, phase)

    def get_dataset(self, **kwargs):
        return CityscapesDataset(self._data_dir, subset=self._phase, **kwargs)
