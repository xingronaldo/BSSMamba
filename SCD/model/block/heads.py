"""
https://github.com/swz30/MIRNet
"""
import torch
import torch.nn as nn


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(GatedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * self.gated(mask)

        return x


class GatedResidualUp(nn.Module):
    def __init__(self, in_channels):
        super(GatedResidualUp, self).__init__()

        self.residual_up = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1,
                                                            output_padding=1, bias=False),
                                         nn.BatchNorm2d(in_channels),
                                         nn.ReLU(True))


        self.gate = GatedConv2d(in_channels, in_channels // 2)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=True)
                                )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = self.residual_up(x)
        residual = self.gate(residual)
        up = self.up(x)
        out = self.relu(up + residual)
        return out


class GatedResidualUpHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=2, dropout_rate=0.1):
        super(GatedResidualUpHead, self).__init__()

        self.up = nn.Sequential(GatedResidualUp(in_channels),
                                GatedResidualUp(in_channels // 2))
        self.smooth = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(True),
                                    nn.Dropout2d(dropout_rate))
        self.final = nn.Conv2d(in_channels // 4, num_classes, 1)

    def forward(self, x):
        x = self.up(x)
        x = self.smooth(x)
        x = self.final(x)

        return x

class FCNHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs=1, dropout_rate=0.15):
        self.num_convs = num_convs
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        convs = []
        convs.append(nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True)))
        for i in range(num_convs - 1):
            convs.append(nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(inter_channels),
                                       nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)
        self.final = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        out = self.convs(x)
        out = self.final(out)

        return out