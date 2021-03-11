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
import shutil

import cv2
import torch
from tqdm import tqdm
import numpy as np

from data.cityscapes import label_ids, label_names
from evaluation import eval_map
from utils import decode


def eval_outputs(data_cfg, dataset, eval_dataloader, model, epoch, decode_cfg, device, logger, metrics):
    decode.device = device
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
        inputs = inputs.to(device)
        # forward the models and loss
        with torch.no_grad():
            outputs = model(inputs)
            dets, instance_maps, det_boxes = decode.decode_output(inputs, model, outputs, infos, decode_cfg, device)
        del inputs
        torch.cuda.empty_cache()

        if "box" in metrics:
            # detections
            for i in range(len(det_boxes)):
                det = det_boxes[i]
                det_result = []
                for j in range(1, data_cfg.num_classes):
                    boxes = np.zeros((0, 5))
                    for k, box in enumerate(det["rois"]):
                        if det["class_ids"][k] == j:
                            boxes = np.append(boxes, np.append(box, det["scores"][k]).reshape((1, 5)), axis=0)
                    det_result.append(boxes)
                det_results.append(det_result)

            # annotations
            for i in range(len(targets[0])):
                class_map, instance_map = targets[0][i], targets[1][i]
                class_tensor = torch.from_numpy(class_map)
                instance_tensor = torch.from_numpy(instance_map)

                annotations = np.zeros((0, 5))
                pre_instance_ids = instance_tensor.unique()
                pre_instance_ids = pre_instance_ids[pre_instance_ids != 0].cpu().numpy()
                for o_j, instance_id in enumerate(pre_instance_ids):
                    mask = instance_tensor == instance_id
                    labels = class_tensor[mask].unique().cpu()
                    assert len(labels) == 1
                    instance_points = mask.nonzero()
                    lt = instance_points.min(0)[0].cpu().numpy()[::-1]
                    rb = instance_points.max(0)[0].cpu().numpy()[::-1]
                    annotation = np.zeros((1, 5))
                    annotation[0, 0:2] = lt
                    annotation[0, 2:4] = rb
                    annotation[0, 4] = labels[0] - 2
                    annotations = np.append(annotations, annotation, axis=0)

                det_annotations.append({
                    "bboxes": annotations[:, :4],
                    "labels": annotations[:, 4]
                })

        if "instance" in metrics:
            for i in range(len(dets)):
                im_name = infos[i][0]
                img_size = infos[i][1]
                instance_map = instance_maps[i]

                basename = os.path.splitext(os.path.basename(im_name))[0]
                txtname = os.path.join(output_dir, basename + '.txt')
                with open(txtname, 'w') as fid_txt:
                    for j in range(1, data_cfg.num_classes):
                        clss = label_names[j]
                        clss_id = label_ids[j]

                        for k in range(len(dets[i])):
                            center_cls, center_conf, instance_id = dets[i][k]
                            if center_cls != j:
                                continue
                            score = center_conf
                            mask = instance_map == instance_id
                            pngname = os.path.join(basename + '_' + clss + '_{}.png'.format(k))
                            # write txt
                            fid_txt.write('{} {} {}\n'.format(pngname, clss_id, score))
                            # save mask
                            cv2.imwrite(os.path.join(output_dir, pngname), mask * 255)
    if "box" in metrics:
        print("------------------------------------box---------------------------------------")
        print("epoch:", epoch)
        print("config:", decode_cfg)
        print("iou for mAP:", 0.5)
        eval_map(det_results, det_annotations, dataset=dataset, iou_thr=0.5)
        print("iou for mAP:", 0.75)
        eval_map(det_results, det_annotations, dataset=dataset, iou_thr=0.75)
    if "instance" in metrics:
        os.environ['CITYSCAPES_DATASET'] = data_cfg.eval_dir
        os.environ['CITYSCAPES_RESULTS'] = output_dir

        # Load the Cityscapes eval script *after* setting the required env vars,
        # since the script reads their values into global variables (at load time).
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling \
            as cityscapes_eval

        logger.write('Evaluating...')
        cityscapes_eval.main()