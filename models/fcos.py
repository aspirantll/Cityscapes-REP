import torch
from torch import nn
from torch.nn import functional as F

from models.fcosnet.fcos import FCOSModule
from models.fcosnet.fpn import FPN, LastLevelP6P7
from models.fcosnet.resnet import ResNet, group_norm
from utils.utils import variance_scaling_
import numpy as np
from .fcosnet import cfg


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


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
    def __init__(self, in_channel, channels, out_channel):
        super().__init__()
        self.up_conv3 = UpConv(in_channel + channels[1], 128, 128)
        self.up_conv4 = UpConv(128 + channels[0], 64, 64)
        self.up_conv5 = UpConv(64, 32, 32)

        self.header = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, out_channel, kernel_size=1, padding=0)
        )

    def forward(self, x, blocks):
        d3 = self.up_conv3(x, blocks[-4])
        d2 = self.up_conv4(d3, blocks[-5])
        d1 = self.up_conv5(d2)

        return self.header(d1)


class FCOSSeg(nn.Module):
    def __init__(self, num_classes):
        super(FCOSSeg, self).__init__()
        cfg.MODEL.FCOS.NUM_CLASSES = num_classes
        self.body = ResNet(cfg)

        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
            else out_channels
        self.fpn = FPN(
            in_channels_list=[
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
            ],
            out_channels=out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
            top_blocks=LastLevelP6P7(in_channels_p6p7, out_channels),
        )

        channels = [64, 64, 256, 512, 1024]

        self.fcos_header = FCOSModule(cfg, out_channels)
        self.spatial_header = SpatialHead(out_channels, channels, out_channel=2)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        blocks = self.body(inputs)
        features = self.fpn(blocks[2:])

        locations, box_cls, box_regression, centerness, center_embeddings = self.fcos_header(features)
        spatial_out = self.spatial_header(features[0], blocks)
        return spatial_out, locations, box_cls, box_regression, centerness, center_embeddings

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