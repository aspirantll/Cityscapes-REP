import torch
from torch import nn
from torch.nn import functional as F

from .fcosnet import fcos_head
from mmdet.models import build_detector, build_backbone, build_neck, build_head
from utils.utils import variance_scaling_
import numpy as np

model = dict(
    type='FCOS',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSModifyHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_cfg=None,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))


class ChannelGate2d(nn.Module):
    """
    Channel Squeeze module
    """

    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):  # skipcq: PYL-W0221
        module_input = x
        x = self.squeeze(x)
        x = x.sigmoid()
        return module_input * x


class SpatialGate2d(nn.Module):
    """
    Spatial squeeze module
    """

    def __init__(self, channels, reduction=None, squeeze_channels=None):
        """
        Instantiate module
        :param channels: Number of input channels
        :param reduction: Reduction factor
        :param squeeze_channels: Number of channels in squeeze block.
        """
        super().__init__()
        assert reduction or squeeze_channels, "One of 'reduction' and 'squeeze_channels' must be set"
        assert not (reduction and squeeze_channels), "'reduction' and 'squeeze_channels' are mutually exclusive"

        if squeeze_channels is None:
            squeeze_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):  # skipcq: PYL-W0221
        module_input = x
        x = self.avg_pool(x)
        x = self.squeeze(x)
        x = F.relu(x, inplace=True)
        x = self.expand(x)
        x = x.sigmoid()
        return module_input * x


class ChannelSpatialGate2d(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2d(channels, reduction=reduction)

    def forward(self, x):  # skipcq: PYL-W0221
        return self.channel_gate(x) + self.spatial_gate(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, temp_channel, out_channels):
        super(UpConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, temp_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(temp_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(temp_channel, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.sc_gate = ChannelSpatialGate2d(out_channels)

    def forward(self, x, e=None):
        x = self.upsample(x)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sc_gate(x)


class SpatialHead(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.up_conv3 = UpConv(512, 256, 256)
        self.up_conv4 = UpConv(256, 128, 128)
        self.up_conv5 = UpConv(128, 64, 64)

        self.header = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, out_channel, kernel_size=1, padding=0)
        )

    def forward(self, x, e):
        d3 = self.up_conv3(x, e)
        d2 = self.up_conv4(d3)
        d1 = self.up_conv5(d2)

        return self.header(d1)


class FCOSSeg(nn.Module):
    def __init__(self, num_classes):
        super(FCOSSeg, self).__init__()
        model["bbox_head"]["num_classes"] = num_classes
        self.backbone = build_backbone(model["backbone"])
        self.neck = build_neck(model["neck"])
        self.bbox_head = build_head(model["bbox_head"])
        self.spatial_header = SpatialHead(out_channel=2)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        blocks = self.backbone(inputs)
        feats = self.neck(blocks)
        cls_score, bbox_pred, centerness, center_embedding = self.bbox_head(feats)
        spatial_out = self.spatial_header(feats[0], blocks[0])
        return spatial_out, cls_score, bbox_pred, centerness, center_embedding

    def init_weight(self):
        for name, module in self.spatial_header.named_modules():
            is_conv_layer = isinstance(module, nn.Conv2d)

            if is_conv_layer:
                if "conv_list" or "header" in name:
                    variance_scaling_(module.weight.data)
                else:
                    nn.init.kaiming_uniform_(module.weight.data)

                if module.bias is not None:
                    if "classifier.header" in name:
                        bias_value = -np.log((1 - 0.01) / 0.01)
                        torch.nn.init.constant_(module.bias, bias_value)
                    else:
                        module.bias.data.zero_()

        variance_scaling_(self.bbox_head.conv_center_embedding.weight.data)
        self.bbox_head.conv_center_embedding.bias.data.zero_()