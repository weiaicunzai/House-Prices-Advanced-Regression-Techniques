import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

#from models.backbones.resnet import resnet50d
import models.backbones.resnet as resnet
#import models.head.fcnhead as fcnhead
from models.head.fcnhead import FCNHead, UpSample2d
from models.dual_gcn import DualGCNHead

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TG(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        #self.head = FCNHead()
        #self.cls_head = UpSample2d(
        #    320,
        #    num_classes,
        #    scale_factor=2
        #)
        self.head = DualGCNHead(
            2048,
            512,
            num_classes
        )
        #self.cls_head = nn.Sequential(
        #    BasicConv2d(
        #        in_channels=2048,
        #        out_channels=512,
        #        kernel_size=1
        #    ),
        #    BasicConv2d(512, num_classes, 1)
        #)

        #self.aux_head = BasicConv2d(
        #    BasicConv2d(1024, 512),
        #    BasicConv2d(512, num_classes)
        #)


    def forward(self, x):
        #print(x.shape)

        B, C, H, W = x.shape

        #assert int(H) % 32 == 0
        #assert int(W) % 32 == 0

        feats = self.backbone(x)
        #out = self.head(feats)
        #out = self.cls_head(out)


        #output = self.cls_head(feats[-1]) # layer 4
        output = self.head(feats[-1]) # layer 4
        output = F.interpolate(
            output,
            size=(H, W),
            align_corners=True,
            mode='bilinear'
        )


        #assert out.shape[2:] == x.shape[2:]
        #assert torch.equal(out.shape[2:], x.shape[2:])


        return output


def tg(num_classes):
    backbone = resnet.resnet50d()
    backbone.fc = nn.Identity()
    backbone.global_pool = nn.Identity()
    #print(backbone)
    net = TG(backbone=backbone, num_classes=num_classes)
    return net


#net = tg(2)
#print(sum(p.numel() for p in net.parameters() if p.requires_grad))
#print(sum(p.numel() for p in net.backbone.parameters() if p.requires_grad))
##print(sum([p.numel() ]))
#img = torch.randn(3, 3, 480, 480)
#
#output = net(img)
#print(output.shape)

#for x in output:
    #print(x.shape)
