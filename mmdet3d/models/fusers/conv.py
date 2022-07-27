from typing import List

import torch
from torch import nn
from mmcv.cnn import normal_init
from mmdet3d.models.builder import FUSERS
from mmcv.runner import BaseModule

__all__ = ["RadarConvFuser", "ConvFuser"]


@FUSERS.register_module()
class RadarConvFuser(BaseModule):
    def __init__(self, in_channels: int, out_channels: int, deconv_blocks: int) -> None:
        super(RadarConvFuser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        deconv = []
        deconv_in = [sum(in_channels) + out_channels]
        deconv_out = [out_channels]
        for i in range(deconv_blocks - 1):
            deconv_in.append(out_channels)
            deconv_out.append(out_channels)
        for i in range(deconv_blocks):
            deconv.append(nn.Sequential(
                nn.Conv2d(deconv_in[i], deconv_out[i], 3, padding=1, bias=False),
                nn.BatchNorm2d(deconv_out[i]),
                nn.ReLU(True))
            )
        self.deconv = nn.ModuleList(deconv)

    def init_weights(self):
        super().init_weights()
        normal_init(self.fuse_conv, mean=0, std=0.001)
        for i in enumerate(self.deconv):
            normal_init(i, mean=0, std=0.001)    

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        res = torch.cat(inputs, dim=1)
        res2 = res.clone()
        out = self.fuse_conv(res)
        out = torch.cat([out, res2], dim=1)
        for layer in self.deconv:
            out = layer(out)
        return out

@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))
