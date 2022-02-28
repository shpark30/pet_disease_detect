from .resnet import ResNet101
from .xception import AlignedXception
from .mobilenet import MobileNetV2

def build_backbone(backbone, output_stride, BatchNorm, pretrained):
    if backbone == 'resnet':
        return ResNet101(output_stride, BatchNorm, pretrained)
    elif backbone == 'xception':
        return AlignedXception(output_stride, BatchNorm, pretrained)
    elif backbone == 'mobilenet':
        return MobileNetV2(output_stride, BatchNorm, pretrained)
    else:
        raise NotImplementedError