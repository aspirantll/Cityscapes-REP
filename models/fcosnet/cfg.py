# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN

MODEL = CN()

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
MODEL.BACKBONE.CONV_BODY = "R-50-FPN-RETINANET"

# Add StopGrad at a specified stage so the bottom layers are frozen
MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
# GN for backbone
MODEL.BACKBONE.USE_GN = False


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
MODEL.FPN = CN()
MODEL.FPN.USE_GN = False
MODEL.FPN.USE_RELU = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
MODEL.GROUP_NORM.EPSILON = 1e-5



MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
MODEL.RESNETS.RES5_DILATION = 1

MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256
MODEL.RESNETS.RES2_OUT_CHANNELS = 256
MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# Deformable convolutions
MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
MODEL.RESNETS.WITH_MODULATED_DCN = False
MODEL.RESNETS.DEFORMABLE_GROUPS = 1

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
MODEL.FCOS = CN()
MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
MODEL.FCOS.PRIOR_PROB = 0.01
MODEL.FCOS.INFERENCE_TH = 0.05
MODEL.FCOS.NMS_TH = 0.6
MODEL.FCOS.PRE_NMS_TOP_N = 1000

# Focal loss parameter: alpha
MODEL.FCOS.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
MODEL.FCOS.LOSS_GAMMA = 2.0

# the number of convolutions used in the cls and bbox tower
MODEL.FCOS.NUM_CONVS = 4

# if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
MODEL.FCOS.CENTER_SAMPLING_RADIUS = 1.5
# IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
MODEL.FCOS.IOU_LOSS_TYPE = "giou"

MODEL.FCOS.NORM_REG_TARGETS = True
MODEL.FCOS.CENTERNESS_ON_REG = True

MODEL.FCOS.USE_DCN_IN_TOWER = True

# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
MODEL.RETINANET.OCTAVE = 2.0
MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
MODEL.RETINANET.USE_C5 = False

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
MODEL.FBNET = CN()
MODEL.FBNET.ARCH = "default"
# custom arch
MODEL.FBNET.ARCH_DEF = ""
MODEL.FBNET.BN_TYPE = "bn"
MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
MODEL.FBNET.WIDTH_DIVISOR = 1
MODEL.FBNET.DW_CONV_SKIP_BN = True
MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
MODEL.FBNET.RPN_HEAD_BLOCKS = 0
MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
SOLVER = CN()
SOLVER.MAX_ITER = 40000

SOLVER.BASE_LR = 0.001
SOLVER.BIAS_LR_FACTOR = 2
# the learning rate factor of deformable convolution offsets
SOLVER.DCONV_OFFSETS_LR_FACTOR = 1.0

SOLVER.MOMENTUM = 0.9

SOLVER.WEIGHT_DECAY = 0.0005
SOLVER.WEIGHT_DECAY_BIAS = 0

SOLVER.GAMMA = 0.1
SOLVER.STEPS = (30000,)

SOLVER.WARMUP_FACTOR = 1.0 / 3
SOLVER.WARMUP_ITERS = 500
SOLVER.WARMUP_METHOD = "linear"

SOLVER.CHECKPOINT_PERIOD = 2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
SOLVER.IMS_PER_BATCH = 16


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
TEST = CN()
TEST.EXPECTED_RESULTS = []
TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
TEST.IMS_PER_BATCH = 8
# Number of detections per image
TEST.DETECTIONS_PER_IMG = 100


# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ---------------------------------------------------------------------------- #
TEST.BBOX_AUG = CN()

 # Enable test-time augmentation for bounding box detection if True
TEST.BBOX_AUG.ENABLED = False

 # Horizontal flip at the original scale (id transform)
TEST.BBOX_AUG.H_FLIP = False

 # Each scale is the pixel size of an image's shortest side
TEST.BBOX_AUG.SCALES = ()

 # Max pixel size of the longer side
TEST.BBOX_AUG.MAX_SIZE = 4000

 # Horizontal flip at each scale
TEST.BBOX_AUG.SCALE_H_FLIP = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
OUTPUT_DIR = "."

PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
