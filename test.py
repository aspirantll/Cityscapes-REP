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
import torch
import os
import cv2
import numpy as np
import warnings

from utils.visualizer import Visualizer

warnings.filterwarnings("ignore")

from utils.logger import Logger
from models import build_model
import data
from utils import post_processor
from utils import image
from configs import Config, Configer
from utils.tranform import CommonTransforms


# global torch configs for training
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
use_cuda = torch.cuda.is_available()
device_type ='cuda' if use_cuda else 'cpu'
device = torch.device(device_type)

# load arguments
print("loading the arguments...")
parser = argparse.ArgumentParser(description="test")
# add arguments
parser.add_argument("--cfg_path", help="the file of cfg", dest="cfg_path", default="./configs/test_cfg.yaml", type=str)
# parse args
args = parser.parse_args()

cfg = Config(args.cfg_path)
data_cfg = cfg.data
decode_cfg = Config(cfg.decode_cfg_path)
trans_cfg = Configer(configs=cfg.trans_cfg_path)

decode_cfg.model_type = cfg.model_type
post_processor.base_dir = data_cfg.save_dir

if data_cfg.num_classes == -1:
    data_cfg.num_classes = data.get_cls_num(data_cfg.dataset)

# validate the arguments
print("test dir:", data_cfg.test_dir)
if data_cfg.test_dir is not None and not os.path.exists(data_cfg.test_dir):
    raise Exception("the test dir cannot be found.")

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


def load_state_dict(model):
    """
    if save_dir contains the checkpoint, then the model will load lastest weights
    :param model:
    :param save_dir:
    :return:
    """
    checkpoint = torch.load(cfg.weights_path, map_location=device_type)
    model.load_state_dict(checkpoint["state_dict"])
    logger.write("loaded the weights:" + cfg.weights_path)


def post_handle(det, instance_map, info):
    img_path = info.img_path
    name = os.path.basename(img_path)

    det_num = len(det[0])
    logger.write("in {} detected {} objs".format(name, det_num))

    if det_num == 0:
        return

    img = cv2.imread(img_path)
    visualizer = Visualizer(img[:, :, ::-1])
    vis_output = visualizer.draw_instance_predictions(det, instance_map)

    save_path = os.path.join(data_cfg.save_dir, name)
    vis_output.save(save_path)
    logger.write("detected result saved in {}".format(save_path))


def handle_output(inputs, infos, model):
    # forward the models and loss
    with torch.no_grad():
        outputs = model(inputs)
        dets, instance_maps = post_processor.decode_output(inputs, model, outputs, decode_cfg, device)
        for i in range(len(dets)):
            if len(dets[i]) != 0:
                post_handle(dets[i], instance_maps[i], infos[i])


def test():
    """
    train the model by the args
    :param args:
    :return:
    """
    # initialize model
    model = build_model(cfg)
    load_state_dict(model)
    model = model.to(device)

    # test model
    model.eval()
    transforms = CommonTransforms(trans_cfg, "val", device)

    post_processor.device = device
    if data_cfg.test_dir is not None:
        # initialize the dataloader by dir
        test_dataloader = data.get_dataloader(data_cfg.batch_size, data_cfg.dataset, data_cfg.test_dir,
                                              with_label=False, phase="test", transforms=transforms)
        # foreach the images
        for iter_id, test_data in enumerate(test_dataloader):
            # to device
            inputs, infos = test_data
            handle_output(inputs, infos, model)
    else:
        img_path = data_cfg.test_image
        input_img = image.load_rgb_image(img_path)
        input, _, info = transforms(input_img, img_path=img_path)
        handle_output(input.unsqueeze(0), [info], model)
    logger.close()


if __name__ == "__main__":
    # train
    logger.write("start to test...")
    test()
