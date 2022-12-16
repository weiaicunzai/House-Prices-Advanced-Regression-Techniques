import os
import sys

sys.path.append(os.getcwd())
import torch
from torch import nn
import torch.nn.functional as F
import models.mgl.resnet as models
import numpy as np
from models.mgl.basicnet import MutualNet, ConcatNet


class MGLNet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=1, zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True, args=None, stage=1):
        super(MGLNet, self).__init__()
        assert layers in [50, 101, 152]
        #assert classes == 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.args = args
        models.BatchNorm = BatchNorm
        self.gamma = 1.0

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.dim = 512

        self.pred = nn.Sequential(
            nn.Conv2d(2048, self.dim, kernel_size=3, padding=1, bias=False),
            BatchNorm(self.dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(self.dim, classes, kernel_size=1)
        )

        self.region_conv = self.pred[0:4] # 2048 -> 512
        self.edge_cat = ConcatNet(BatchNorm) # concat low-level feature map to predict edge

        # cascade mutual net
        self.mutualnet0 = MutualNet(BatchNorm, dim=self.dim, num_clusters=32, dropout=dropout)
        # print(args.stage, ';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
        if stage == 1:
            self.mutualnets = nn.ModuleList([self.mutualnet0])
        elif stage == 2:
            self.mutualnet1 = MutualNet(BatchNorm, dim=self.dim, num_clusters=32, dropout=dropout)
            self.mutualnets = nn.ModuleList([self.mutualnet0, self.mutualnet1])

    def forward(self, x, y=None, iter_num=0, y2=None):
        x_size = x.size()
        # print(x_size[2], x_size)
        #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        #h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        #w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0
        h = int((x_size[2]) / 8 * self.zoom_factor)
        w = int((x_size[3]) / 8 * self.zoom_factor)

        # import sys; sys.exit()

        ##step1. backbone layer
        # Multi-Task Feature Extraction (MTFE).
        # fMTFEtakes animage as the input, and
        # produces two task-specific featuremaps —
        # one for COD and the other for COEE.
        #print(x.shape)
        # print(x.shape) 473 x 473
        #print(x.shape)
        x_0 = self.layer0(x) # 119 x 119
        #print(x_0.shape, 55)
        x_1 = self.layer1(x_0) # 60 x 60
        #print(x_1.shape, 44)
        x_2 = self.layer2(x_1) # 60 x 60
        #print(x_2.shape, 33)
        x_3 = self.layer3(x_2) # 60 x 60
        #print(x_3.shape, 22)
        x_4 = self.layer4(x_3) # 60 x 60
        #print(x_4.shape)
        #print(x_4.shape, 11)
        #import sys; sys.exit()
        # print(x_4.mean())

        ##step2. concat edge feature by side-output feature
        # print(x_4.shape, 'x_4 shape')
        #print(x_1.shape, x_2.shape, x_3.shape, x_4.shape)
        # print(x_0.device)

        # 60 x 60
        cod_x = self.region_conv(x_4) # 2048 -> 512
        #print(cod_x.shape, 111)
        coee_x = self.edge_cat(x_1, x_2, x_3, x_4, cod_x.shape[2:]) # edge pixel-level feature

        # 60 x 60
        # print(cod_x.shape, coee_x.shape) bs, 512, 60 x 60,  3, 512, 60, 60

        # print(coee_x.shape, cod_x.shape, 111)

        main_loss = 0.
        #print(len(self.mutualnets), ';;;;;;;')
        for net in self.mutualnets:
            # print(net)
            #print(net)
            # net(edge, region)
            #print(coee_x.shape, cod_x.shape)
            n_coee_x, coee, n_cod_x, cod = net(coee_x, cod_x)
            #print(net, 'ccccccccccccc')

            # print(n_coee_x.mean())
            # print(coee.mean())
            # print(n_cod_x.mean())
            # print(cod_x.mean())
            coee_x = coee_x + n_coee_x
            cod_x = cod_x + n_cod_x

            # print(self.zoom_factor)
            # import sys; sys.exit()
            if self.zoom_factor != 1:  # zoom_factor = 8
                coee = F.interpolate(coee, size=(h, w),  mode='bilinear', align_corners=True)
                cod = F.interpolate(cod, size=(h, w), mode='bilinear', align_corners=True)
                # print(self.gamma)
                #if self.training:
                #    # print(coee, y2)
                #    # print(torch.unique(y2))
                #    # coee.
                #    # coee = torch.sigmoid(coee)
                #    # print(torch.unique(y2))
                #    main_loss += self.gamma * self.criterion(coee, y2) # supervise edge
                #    # print(main_loss, 111)
                #    # print(coee.shape, y2.shape)
                #    main_loss += self.criterion(cod, y) # supervise region
                #    # import sys; sys.exit()

        #import sys; sys.exit()
        #if self.training:
        #    return cod, coee, main_loss
        #else:
        #    return cod, coee
        #print(cod.shape, coee.shape)

        output = torch.cat([cod, coee], dim=1)
        #output = self.pred(output)


        return output

        #return tor

def mgl(num_classes):
    #model = MGLNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, BatchNorm=BatchNorm, pretrained=False, args=args).cuda()

    net = MGLNet(layers=50, classes=num_classes, zoom_factor=8, stage=1, pretrained=False)

    return net


#img = torch.randn(3, 3, 8 * 10, 8 * 10)
#img = torch.randn(3, 3, 8 * 60, 8 * 60)
#net = mgl(2)
#output = net(img)
#print(output.shape)

#print(net)