
import torch
import torch.nn as nn

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpSample2d(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2.0):

        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = BasicConv2d(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return x


class FCNHead(nn.Module):
    def __init__(self):
        super().__init__()
        #self.in_channels = in_channels
        #self.num_classes = num_classes

        self.up1 = UpSample2d(64, 64, scale_factor=1)
        self.up2 = UpSample2d(256, 64, scale_factor=2)
        self.up3 = UpSample2d(512, 64, scale_factor=4)
        self.up4 = UpSample2d(1024, 64, scale_factor=4)
        self.up5 = UpSample2d(2048, 64, scale_factor=4)
        #self.up4 = UpSample2d(1024, 64, scale_factor=8)
        #self.up5 = UpSample2d(2048, 64, scale_factor=16)

        #self.upsample = nn.Sequential(
        #    BasicConv2d(128, 64),
        #    BasicConv2d(64, 64)
        #)

        #self.up2 = nn.Sequential(
        #    UpSample2d(),
        #)


    def forward(self, feats):
        for feat in feats:
            print(feat.shape)
        out = []
        out.append(self.up1(feats[0]))
        out.append(self.up2(feats[1]))
        out.append(self.up3(feats[2]))
        out.append(self.up4(feats[3]))
        out.append(self.up5(feats[4]))

        out = torch.cat(out, dim=1)

        return out


        #for feat in feats:
            #feat.append()
