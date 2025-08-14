import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mamba import MambaBlk


#Difference feature refinement
class Decoder(nn.Module):
    def __init__(self, channels, depth=2):
        super(Decoder, self).__init__()
        self.network = nn.ModuleList()
        for _ in range(depth):
            self.network.append(MambaBlk(channels))

    def forward(self, x):
        #BCHW->BCHW
        x = rearrange(x, 'b c h w -> b h w c')
        for layer in self.network:
            x = layer(x)
        x = rearrange(x, 'b h w c -> b c h w')

        return x
