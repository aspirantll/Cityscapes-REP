__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import argparse
import data
import os
import os.path as op
import multiprocessing
from tqdm import tqdm

from configs import Config
from utils.label_io import save_labels
from data.cityscapes import CityscapesDataset


# load arguments
print("loading the arguments...")
parser = argparse.ArgumentParser(description="training")
# add arguments
parser.add_argument("--cfg_path", help="the file of cfg", dest="cfg_path", default="./configs/train_cfg.yaml", type=str)
parser.add_argument("--phases", help="which phase", dest="phases", default=["train"], type=list)
parser.add_argument("--workers", help="the num of worker process", dest="workers", default=4, type=list)

# parse args
args = parser.parse_args()

cfg = Config(args.cfg_path)
data_cfg = cfg.data

if isinstance(data_cfg.input_size, str):
    data_cfg.input_size = eval(data_cfg.input_size)

if data_cfg.num_classes == -1:
    data_cfg.num_classes = data.get_cls_num(data_cfg.dataset)


def process_item(info_queue, dataset, target_dir, j, cover, filter_empty):
    filename = op.basename(dataset.filenames[j])[:-4] + ".npz"
    save_path = op.join(target_dir, filename)
    if cover or not op.exists(save_path):
        input_tensor, label, _ = dataset[j]
        if not filter_empty or len(label[0])!=0:
            save_labels(input_tensor, label, save_path)
    info_queue.put(j)


def preprocess(cover=False, filter_empty=True):
    for i in range(len(args.phases)):
        subset = args.phases[i]
        dataset = CityscapesDataset(data_cfg.train_dir, data_cfg.input_size, subset=subset)
        target_dir = op.join(data_cfg.train_dir, 'preprocessed/' + subset)
        if not op.exists(target_dir):
            os.makedirs(target_dir)

        pool = multiprocessing.Pool(processes=args.workers)
        info_queue = multiprocessing.Manager().Queue()
        n = len(dataset)
        for j in range(n):
            pool.apply_async(process_item, args=(info_queue, dataset, target_dir, j, cover, filter_empty))
        pool.close()
        for j in tqdm(range(n), desc=subset):
            j = info_queue.get()
        pool.join()


if __name__ == "__main__":
    preprocess()