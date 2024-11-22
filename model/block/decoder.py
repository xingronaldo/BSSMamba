import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mamba import LCPMamba


#Difference feature refinement
class Decoder(nn.Module):
    def __init__(self, channels, depth=2):
        super(Decoder, self).__init__()
        self.network = nn.ModuleList()
        for _ in range(depth):
            self.network.append(SpectralMamba(channels))

    def forward(self, x):
        #BCHW->BCHW
        for layer in self.network:
            x = layer(x)

        return x


class SpectralMamba(nn.Module):
    def __init__(self, channels):
        super(SpectralMamba, self).__init__()
        self.proj = nn.Linear(channels, channels, bias=False)
        self.fre = SpectralTransform(channels)
        self.norm = nn.LayerNorm(channels)
        self.attn = LCPMamba(channels)

    def forward(self, x):
        # BCHW->BCHW
        x_norm = rearrange(x, 'b c h w -> b h w c')
        x_norm = self.norm(x_norm)
        spa = self.attn(x_norm)
        fre = self.proj(x_norm)
        fre = rearrange(fre, 'b h w c -> b c h w')
        fre = self.fre(fre)

        spa = rearrange(spa, 'b h w c -> b c h w')
        out = fre * spa + x

        return out


class SpectralTransform(nn.Module):
    def __init__(self, channels):
        super(SpectralTransform, self).__init__()
        self.fu = FourierUnit(channels, channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.fu(x)
        out = self.out_conv(out)

        return out


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.size()

        ffted = torch.fft.rfftn(x, s=(H, W), dim=(-2, -1), norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (b, c, h, w/2+1, 2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (b, c, 2, h, w/2+1)
        ffted = ffted.view((B, -1,) + ffted.size()[3:])

        ffted = self.conv(ffted)  # (batch, c*2, h, w/2+1)
        ffted = F.silu(ffted)
        ffted = ffted.view((B, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        x = torch.fft.irfftn(ffted, s=(H, W), dim=(-2, -1), norm='ortho')

        return x





















class CrossAttn(nn.Module):
    def __init__(self, channels):
        super(CrossAttn, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.pos = ScaledSinuEmbedding(channels)
        self.crossattn = MultiScaleDeformableAttention(embed_dims=channels, num_levels=1, num_heads=4,
                                                       num_points=4, batch_first=True, dropout=0.)

    def get_deform_inputs(self, x1, x2):
        _, _, H1, W1 = x1.size()
        _, _, H2, W2 = x2.size()
        spatial_shapes = torch.as_tensor([(H2, W2)], dtype=torch.long, device=x2.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = get_reference_points([(H1, W1)], x1.device)

        return reference_points, spatial_shapes, level_start_index

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        reference_points, spatial_shapes, level_start_index = self.get_deform_inputs(x1, x2)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        query_pos = self.pos(x1)
        x = self.crossattn(query=self.norm1(x1), value=self.norm2(x2), identity=x2,
                           reference_points=reference_points, spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index, query_pos=query_pos)

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]

    return reference_points


class SelfAttnBlock(nn.Module):
    def __init__(self, channels):
        super(SelfAttnBlock, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.attn = MultiScaleDeformableAttention(embed_dims=channels, num_levels=1, num_heads=4,
                                                  num_points=4, batch_first=True, dropout=0.)
        self.ffn = nn.ModuleList([])
        self.ffn.append(nn.ModuleList([nn.Conv2d(channels, channels, 3, stride=1, padding=1, groups=channels, bias=True),
                                 nn.Conv2d(channels // 2, channels, 1, bias=True)]))

    def get_deform_inputs(self, x):
        _, _, H, W = x.size()
        spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = get_reference_points([(H, W)], x.device)

        return reference_points, spatial_shapes, level_start_index

    def forward(self, x):
        B, C, H, W = x.shape

        reference_points, spatial_shapes, level_start_index = self.get_deform_inputs(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attn(query=self.norm(x), value=self.norm(x), identity=x,
                      reference_points=reference_points, spatial_shapes=spatial_shapes,
                      level_start_index=level_start_index, query_pos=None)

        x_res = x.clone()
        x = self.ffn_norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x_res = rearrange(x_res, 'b (h w) c -> b c h w', h=H, w=W)
        for dwconv, prj in self.ffn:
            x1, x2 = dwconv(x).chunk(2, dim=1)
            x = F.silu(x1) * x2
            x = prj(x)
        x = x + x_res

        return x



