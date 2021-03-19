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
from torch.utils import data
from data.dataset import DatasetBuilder
from utils.image import load_rgb_image
from utils.tranform import TransInfo


class DirDataset(data.Dataset):
    """
    the dataset for image dir.
    """
    def __init__(self, data_dir, transforms=None, from_file=False):
        # save the parameters
        self._data_dir = data_dir
        if transforms is not None:
            self._transforms = transforms  # ADDED THIS
        # scan the dir
        self.imgs = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in
                                fn if f.endswith(r".jpg") or f.endswith(r".png")]

    def __getitem__(self, index):
        # locating index
        path = self.imgs[index]
        input_img = load_rgb_image(path)
        img_size = input_img.shape[:2]
        if self._transforms is not None:
            input_img, _ = self._transforms(input_img)
        return input_img, None, TransInfo(path, img_size)

    def __len__(self):
        return len(self.imgs)


class DirDatasetBuilder(DatasetBuilder):
    def __init__(self, data_dir,  phase):
        super().__init__(data_dir, phase)

    def get_dataset(self, **kwargs):
        return DirDataset(self._data_dir, **kwargs)