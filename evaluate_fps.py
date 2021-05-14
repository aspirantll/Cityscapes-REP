from __future__ import print_function


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
from time import time

import torch
import os
import numpy as np
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

import data
from models import build_model

from configs import Config, Configer
from utils.logger import Logger
from utils import post_processor
from utils.tranform import CommonTransforms

# global torch configs for training
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
use_cuda = torch.cuda.is_available()
device_type ='cuda' if use_cuda else 'cpu'
device = torch.device(device_type)

post_processor.device = device
post_processor.draw_flag = False


# load arguments
print("loading the arguments...")
parser = argparse.ArgumentParser(description="test")
# add arguments
parser.add_argument("--cfg_path", help="the file of cfg", dest="cfg_path", default="./configs/eval_cfg.yaml", type=str)
# parse args
args = parser.parse_args()

cfg = Config(args.cfg_path)
data_cfg = cfg.data
decode_cfg = Config(cfg.decode_cfg_path)
trans_cfg = Configer(configs=cfg.trans_cfg_path)

decode_cfg.model_type = cfg.model_type


if data_cfg.num_classes == -1:
    data_cfg.num_classes = data.get_cls_num(data_cfg.dataset)
# validate the arguments
print("eval dir:", data_cfg.eval_dir)
if data_cfg.eval_dir is not None and not os.path.exists(data_cfg.eval_dir):
    raise Exception("the eval dir cannot be found.")

print("save dir:", data_cfg.save_dir)
if not os.path.exists(data_cfg.save_dir):
    os.makedirs(data_cfg.save_dir)

# set seed
np.random.seed(cfg.seed)
torch.random.manual_seed(cfg.seed)
if use_cuda:
    torch.cuda.manual_seed_all(cfg.seed)

Logger.init_logger(data_cfg)
logger = Logger.get_logger()


def load_state_dict(model, weights_path):
    """
    if save_dir contains the checkpoint, then the model will load lastest weights
    :param model:
    :param save_dir:
    :return:
    """
    checkpoint = torch.load(weights_path, map_location=device_type)
    model.load_state_dict(checkpoint["state_dict"])
    logger.write("loaded the weights:" + weights_path)
    return checkpoint["epoch"]


def evaluate_fps(test_dataloader, weights_path):
    """
    validate model for a epoch
    :param transforms:
    :param eval_dataloader:
    :return:
    """
    # initialize
    model = build_model(cfg)
    epoch = load_state_dict(model, weights_path)
    model = model.to(device)

    model.eval()
    num_iter = len(test_dataloader)

    start_time = time()
    # foreach the images
    for iter_id, eval_data in tqdm(enumerate(test_dataloader), total=num_iter,
                                   desc="eval fps for epoch {}".format(epoch)):
        inputs, infos = eval_data
        # forward the models and loss
        with torch.no_grad():
            outputs = model(inputs)
            post_processor.decode_output(inputs, model, outputs, decode_cfg, device)
        del inputs
        torch.cuda.empty_cache()
    end_time = time()
    print("total times: %f" % (end_time-start_time))
    print("total image num: %d" % num_iter)
    print("fps: %f" % (num_iter/(end_time-start_time)))


if __name__ == "__main__":
    data_cfg.batch_size = 1
    transforms = CommonTransforms(trans_cfg, "val", device)
    test_dataloader = data.get_dataloader(data_cfg.batch_size, data_cfg.dataset, data_cfg.eval_dir,
                                          with_label=False, phase=data_cfg.subset, transforms=transforms)

    # eval
    print("start to evaluate fps...")
    evaluate_fps(test_dataloader, cfg.weights_path)
    # logger.close()