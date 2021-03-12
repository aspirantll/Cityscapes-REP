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


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


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
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.upsample = UpsamplerBlock(in_channels, out_channels)
        self.conv1 = non_bottleneck_1d(out_channels, 1)
        self.conv2 = non_bottleneck_1d(out_channels, 1)
        self.sc_gate = ChannelSpatialGate2d(out_channels)

    def forward(self, x, e=None):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sc_gate(x)


class SpatialHead(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.up_conv3 = UpConv(256, 128)
        self.up_conv4 = UpConv(128, 64)
        self.up_conv5 = UpConv(64, 32)

        self.header = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, out_channel, kernel_size=1, padding=0)
        )

    def forward(self, x, e):
        d3 = self.up_conv3(x)
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