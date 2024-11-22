import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(FPN, self).__init__()

        self.p2 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.p3 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.p4 = nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)
        self.p5 = nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels, kernel_size=1)

        self.p2_smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        c2, c3, c4, c5 = input

        p5 = self.p5(c5)
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.p4(c4) + p5_up
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.p3(c3) + p4_up
        p3_up = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.p2(c2) + p3_up

        p2 = self.p2_smooth(p2)

        return p2




