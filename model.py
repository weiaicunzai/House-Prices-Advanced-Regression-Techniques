import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, input_channels,
                       out_channels,
                       kernel_size,
                       stride=1,
                       padding=1):

        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class CRNN(nn.Module):

    def __init__(self, input_channels=1, num_classes=1):

        super().__init__()
        self.conv1 = BasicConv(input_channels, 64, 3)
        self.conv2 = BasicConv(64, 128, 3)
        self.conv3 = BasicConv(128, 256, 3)
        self.conv4 = BasicConv(256, 256, 3)
        self.conv5 = BasicConv(256, 512, 3)
        self.conv6 = BasicConv(512, 512, 3)
        self.conv7 = BasicConv(512, 512, 3)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.symmetric_maxpool = nn.MaxPool2d((2, 2), (2, 2))
        self.asymmetric_maxpool = nn.MaxPool2d((2, 2), (2, 1), (0, 1))


    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.symmetric_maxpool(x)
        x = self.conv2(x)
        x = self.symmetric_maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.asymmetric_maxpool(x)
        x = self.conv5(x)
        x = self.bn1(x)
        x = self.conv6(x)
        x = self.bn2(x)
        x = self.asymmetric_maxpool(x)
        x = self.conv7(x)

        return x

a = torch.Tensor(32, 1, 32, 100)
crnn = CRNN()
print(sum(p.numel() for p in crnn.parameters()))
print(a.shape)
print(crnn(a).shape)
