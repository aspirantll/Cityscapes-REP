import torch
from mmcv.cnn import normal_init
from torch import nn
from torch.nn import functional as F

from configs import Config
from utils.target_generator import generate_fcos_annotations
from mmdet.models import build_backbone, build_neck, build_head
from utils.utils import variance_scaling_, generate_coordinates, zero_tensor, convert_corner_to_corner
import numpy as np

from .lovasz_losses import lovasz_hinge

model = dict(
    type='FCOS',
    pretrained=None,
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256,
        stride=2,
        num_outs=5),
    bbox_head=dict(
        type='FCOSHead',
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
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))


train_cfg = Config(cfg=dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False))
test_cfg = Config(cfg=dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.2,
    nms=dict(type='nms', iou_threshold=0.2),
    max_per_img=100))



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
        model["bbox_head"].update(train_cfg=train_cfg)
        model["bbox_head"].update(test_cfg=test_cfg)
        self.bbox_head = build_head(model["bbox_head"])
        self.spatial_header = SpatialHead(out_channel=3)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        blocks = self.backbone(inputs)
        feats = self.neck(blocks)
        cls_score, bbox_pred, centerness = self.bbox_head(feats)
        spatial_out = self.spatial_header(feats[0], blocks[0])
        return spatial_out, cls_score, bbox_pred, centerness

    def init_weight(self):
        self.neck.init_weights()
        self.bbox_head.init_weights()

        for name, module in self.spatial_header.named_modules():
            is_conv_layer = isinstance(module, nn.Conv2d)

            if is_conv_layer:
                normal_init(module, std=0.01)




class AnchorFreeAELoss(object):
    def __init__(self, device, weight=1):
        self._device = device
        self._weight = weight
        self._xym = generate_coordinates().to(device)

    def __call__(self, ae, targets):
        """
        :param ae:
        :param targets: (instance_map_list)
        :return:
        """
        # prepare step
        det_annotations, instance_ids_list, instance_map_list = targets
        b, c, h, w = ae.shape

        xym_s = self._xym[:, 0:h, 0:w].contiguous()  # 2 x h x w

        ae_loss = zero_tensor(self._device)
        for b_i in range(b):
            instance_ids = instance_ids_list[b_i]
            instance_map = instance_map_list[b_i]

            n = len(instance_ids)
            if n <= 0:
                continue

            spatial_emb = torch.tanh(ae[b_i, 0:2]) + xym_s  # 2 x h x w
            sigma = ae[b_i, 2:3]  # n_sigma x h x w

            var_loss = zero_tensor(self._device)
            instance_loss = zero_tensor(self._device)

            for o_j, instance_id in enumerate(instance_ids):
                in_mask = instance_map.eq(instance_id).view(1, h, w) # 1 x h x w

                # calculate center of attraction
                o_lt = det_annotations[b_i, o_j, 0:2][::-1].astype(np.int32)
                o_rb = det_annotations[b_i, o_j, 2:4][::-1].astype(np.int32)

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(sigma)]

                s = sigma_in.mean().view(1, 1, 1)  # n_sigma x 1 x 1

                # calculate var loss before exp
                var_loss = var_loss + \
                           torch.mean(
                               torch.pow(sigma_in - s.detach(), 2))
                assert not torch.isnan(var_loss)

                s = torch.exp(s)

                # limit 2*box_size mask
                lt, rb = convert_corner_to_corner(o_lt, o_rb, h, w, 1.5)
                selected_spatial_emb = spatial_emb[:, lt[0]:rb[0], lt[1]:rb[1]]
                label_mask = in_mask[:, lt[0]:rb[0], lt[1]:rb[1]].float()
                center_index = ((o_lt + o_rb) / 2).astype(np.int32)
                center = xym_s[:, center_index[0], center_index[1]].view(2, 1, 1)
                # calculate gaussian
                dist = torch.exp(-1 * torch.sum(
                    torch.pow(selected_spatial_emb - center, 2) * s, 0, keepdim=True))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                                lovasz_hinge(dist * 2 - 1, label_mask)

            ae_loss += (var_loss + instance_loss) / max(n, 1)
        # compute mean loss
        return ae_loss / b


class FCOSLoss(nn.Module):
    def __init__(self, device):
        super(FCOSLoss, self).__init__()
        self._device = device
        self._loss_names = ["cls_loss", "wh_loss", "center_loss", "ae_loss", "total_loss"]
        self.ae_loss = AnchorFreeAELoss(device)

    def forward(self, model, outputs, targets):
        # unpack the output
        spatial_out, cls_scores, bbox_preds, centernesses = outputs
        gt_boxes, gt_labels, det_annotations, instance_ids_list, instance_map_list = generate_fcos_annotations(spatial_out.shape, targets, self._device)
        det_losses = model.bbox_head.loss(cls_scores, bbox_preds, centernesses, gt_boxes, gt_labels, None)

        losses = []
        losses.extend(det_losses.values())
        losses.append(self.ae_loss(spatial_out, (det_annotations, instance_ids_list, instance_map_list)))

        # compute total loss
        total_loss = torch.stack(losses).sum()
        losses.append(total_loss)

        return total_loss, {self._loss_names[i]: losses[i] for i in range(len(self._loss_names))}

    def get_loss_states(self):
        return self._loss_names