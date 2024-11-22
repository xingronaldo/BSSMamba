import torch
import torch.nn as nn
from .mamba import MambaBlk
from einops import rearrange
import torch.nn.functional as F


# Bi-temporal feature refinement and fusion
class Encoder(nn.Module):
    def __init__(self, channels, depth=2):
        super(Encoder, self).__init__()
        self.network = nn.ModuleList()
        for _ in range(depth):
            self.network.append(MambaBlk(channels))
        self.norm = nn.LayerNorm(channels * 2)
        self.diff_conv = nn.Conv2d(channels * 2, channels, 1, bias=True)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = rearrange(x1, 'b c h w -> b h w c')
        x2 = rearrange(x2, 'b c h w -> b h w c')
        for layer in self.network:
            x1 = layer(x1)
            x2 = layer(x2)

        diff = torch.abs(x1 - x2)
        add = x1 + x2
        x = torch.cat([diff, add], dim=-1)
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w', h=H, w=W)
        out = self.diff_conv(x)

        return out

