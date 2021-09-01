__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import glob
import os
import shutil
from collections import OrderedDict

import cv2
import torch
from tqdm import tqdm
import numpy as np

from evaluation import eval_map
from utils import post_processor


def eval_outputs(data_cfg, dataset, eval_dataloader, model, epoch, decode_cfg, device, logger, metrics):
    post_processor.device = device
    output_dir = os.path.join(data_cfg.save_dir, 'results_' + str(epoch))

    if os.path.exists("./matches.json"):
        os.remove("./matches.json")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # eval
    model.eval()
    num_iter = len(eval_dataloader)

    # foreach the images
    det_results = []
    det_annotations = []
    for iter_id, eval_data in tqdm(enumerate(eval_dataloader), total=num_iter,
                                   desc="eval for epoch {}".format(epoch)):
        # to device
        inputs, targets, infos = eval_data
        # forward the models and loss
        with torch.no_grad():
            outputs = model(inputs)
            dets = post_processor.decode_output(inputs, model, outputs, decode_cfg, infos, device)
        del inputs
        torch.cuda.empty_cache()

        img_num = len(dets)
        for i in range(img_num):
            classes, scores, boxes, instance_ids = dets[i]

            if "box" in metrics:
                # detections
                det_result = []
                for j in range(1, data_cfg.num_classes):
                    det_boxes = np.zeros((0, 5))
                    for k, box in enumerate(boxes):
                        if classes[k] == j:
                            det_boxes = np.append(det_boxes, np.append(box, scores[k]).reshape((1, 5)), axis=0)
                    det_result.append(det_boxes)
                det_results.append(det_result)

                # annotations
                class_ids, _ = targets[i][0], targets[i][1]
                det_annotations.append({
                    "bboxes": (class_ids[:, :4] - np.array((*infos[i].pad_size, *infos[i].pad_size))
                               .reshape(1, 4))/infos[i].scale,
                    "labels": class_ids[:, 4] - 1
                })

    metric_aps = OrderedDict()

    if "box" in metrics:
        print("------------------------------------box---------------------------------------")
        print("epoch:", epoch)
        print("config:", decode_cfg)
        print("iou for mAP:", 0.5)
        metric_aps["Box_AP50"] = eval_map(det_results, det_annotations, dataset=dataset, iou_thr=0.5)
        print("iou for mAP:", 0.75)
        metric_aps["Box_AP75"] = eval_map(det_results, det_annotations, dataset=dataset, iou_thr=0.75)

    return {epoch: metric_aps}