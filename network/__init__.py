__all__ = ["backbone", "sync_batchnorm", "deeplab"]

import torch
from .deeplab import DeepLabv3plus


def build_model(args):
    return DeepLabv3plus(args.img_ch, args.out_ch, backbone=args.backbone, pretrained=args.pretrained, freeze_bn=args.freeze_bn)