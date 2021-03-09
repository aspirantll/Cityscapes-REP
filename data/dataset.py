__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import abc


def is_train_phase(phase):
    return "train" in phase


def is_val_phase(phase):
    return "val" in phase


class DatasetBuilder(object):
    def __init__(self, data_dir, phase="train"):
        if data_dir is None:
            raise Exception("The data_dir must be not None.")
        self._data_dir = data_dir
        self._phase = phase

    @abc.abstractmethod
    def get_dataset(self, **kwargs):
        pass