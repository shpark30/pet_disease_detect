import torch
import torch.nn as nn
import torch.nn.functional as F
from .aspp import build_aspp
from .decoder import build_decoder
from .backbone import build_backbone

class DeepLabv3plus(nn.Module):
    def __init__(self, img_ch=3, out_ch=3, backbone='xception', pretrained=False, freeze_bn=False):
        super(DeepLabv3plus, self).__init__()
        self.out_ch = out_ch
        output_stride = 16
        BatchNorm = nn.BatchNorm2d
        
        # DCNN (XCEPTION)
        self.backbone = build_backbone(backbone, output_stride, nn.BatchNorm2d, pretrained=pretrained)
        # ASPP
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        # Decoder
        self.decoder = build_decoder(out_ch, backbone, BatchNorm)
        self.freeze_bn = freeze_bn
        
    def forward(self, input):
        d, low_level_feat = self.backbone(input)
        x = self.aspp(d)
        x = self.decoder(x, low_level_feat)
        sr = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return sr
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                
                
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

                                
    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
            
        
        
