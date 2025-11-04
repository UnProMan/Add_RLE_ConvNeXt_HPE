import torch
import torch.nn as nn
from timm.models.layers import DropPath

class DeConv(nn.Sequential):
    def __init__(self, inplances, planes, upscale_factor=2, kernel_size = 3, up = True):
        super().__init__()
        size = kernel_size

        if kernel_size == 7: pad = 3
        elif kernel_size == 5: pad = 2
        else: pad = 1

        self.dwconv = nn.Conv2d(inplances, inplances, kernel_size=size, stride=1, padding=pad, groups=inplances)
        self.norm = nn.BatchNorm2d(inplances)
        self.pwconv = nn.Conv2d(inplances, planes, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=upscale_factor) if up else nn.Identity()

    def forward(self, x):
        input = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = self.upsample1(x)

        return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> BatchNorm2d (channels_first) -> 1x1 Conv -> ReLU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); BatchNorm2d (channels_last) -> Linear -> ReLU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU(inplace=True)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = input + self.drop_path(x)
        return x

