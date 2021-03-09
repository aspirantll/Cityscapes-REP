"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="generateCrop")
# add arguments
parser.add_argument("--data_root", help="the root dir of cityscape", dest="data_root", default="", type=str)
parser.add_argument("--crop_size", help="the size of crop", dest="crop_size", default=512, type=int)
parser.add_argument("--subset", help="the subset of crop", dest="subset", default="train", type=str)
parser.add_argument("--target", help="the target of crop", dest="target", default="train0", type=str)
# parse args
args = parser.parse_args()


def collect_obj_ids(instance_np, OBJ_ID):
    object_mask = np.logical_and(instance_np >= OBJ_ID * 1000, instance_np < (OBJ_ID + 1) * 1000)

    ids = np.unique(instance_np[object_mask])
    ids = ids[ids != 0]
    return ids


def collect_ids(instance_np):
    selected_ids = np.zeros(0)
    # car
    ids = collect_obj_ids(instance_np, 26)
    np.random.shuffle(ids)
    if len(ids) < 10:
        selected_ids = np.append(selected_ids, ids)
    else:
        selected_ids = np.append(selected_ids, ids[:10])

    # truck
    ids = collect_obj_ids(instance_np, 27)
    selected_ids = np.append(selected_ids, ids)

    # truck
    ids = collect_obj_ids(instance_np, 28)
    selected_ids = np.append(selected_ids, ids)

    # truck
    ids = collect_obj_ids(instance_np, 31)
    selected_ids = np.append(selected_ids, ids)

    # truck
    ids = collect_obj_ids(instance_np, 32)
    selected_ids = np.append(selected_ids, ids)

    return selected_ids


def process(tup):
    im, inst = tup

    image_path = os.path.splitext(os.path.relpath(im, os.path.join(IMAGE_DIR, CROP_SUBSET)))[0]
    image_path = os.path.join(IMAGE_DIR, CROP_TARGET, image_path)
    instance_path = os.path.splitext(os.path.relpath(inst, os.path.join(INSTANCE_DIR, CROP_SUBSET)))[0]
    instance_path = os.path.join(INSTANCE_DIR, CROP_TARGET, instance_path)

    try:  # can't use 'exists' because of threads
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
    except FileExistsError:
        pass

    image = Image.open(im)
    instance = Image.open(inst)
    w, h = image.size

    # loop over instances
    instance_np = np.array(instance, copy=False)
    ids = collect_ids(instance_np)

    # loop over instances
    for j, id in enumerate(ids):
        y, x = np.where(instance_np == id)
        ym, xm = np.mean(y), np.mean(x)

        ii = int(np.clip(ym - CROP_SIZE / 2, 0, h - CROP_SIZE))
        jj = int(np.clip(xm - CROP_SIZE / 2, 0, w - CROP_SIZE))

        im_crop = image.crop((jj, ii, jj + CROP_SIZE, ii + CROP_SIZE))
        instance_crop = instance.crop((jj, ii, jj + CROP_SIZE, ii + CROP_SIZE))

        im_crop.save(image_path + "_{:03d}.png".format(j))
        instance_crop.save(instance_path + "_{:03d}.png".format(j))


if __name__ == '__main__':
    # cityscapes dataset
    CITYSCAPES_DIR=args.data_root
    CROP_SIZE=args.crop_size
    CROP_SUBSET = args.subset
    CROP_TARGET = args.target

    IMAGE_DIR=os.path.join(CITYSCAPES_DIR, 'leftImg8bit')
    INSTANCE_DIR=os.path.join(CITYSCAPES_DIR, 'gtFine')

    # load images/instances
    images = glob.glob(os.path.join(IMAGE_DIR, CROP_SUBSET, '*/*.png'))
    images.sort()
    instances = glob.glob(os.path.join(INSTANCE_DIR, CROP_SUBSET, '*/*instanceIds.png'))
    instances.sort()
    process((images[0], instances[0]))

    with Pool(8) as p:
        r = list(tqdm(p.imap(process, zip(images,instances)), total=len(images)))
