# Ultralytics YOLO 🚀, AGPL-3.0 license
"""DEIM HGNetv2 backbone building blocks for YAML-level parsing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(act='relu'):
    """Local minimal activation helper for DEIM HGNetv2 blocks."""
    if isinstance(act, nn.Module):
        return act
    if act is None:
        return nn.Identity()
    if not isinstance(act, str):
        raise TypeError(f'Unsupported activation type: {type(act)}')

    act_name = act.lower()
    if act_name == 'relu':
        return nn.ReLU(inplace=True)
    if act_name in {'silu', 'swish'}:
        return nn.SiLU(inplace=True)
    if act_name == 'gelu':
        return nn.GELU()
    raise ValueError(f'Unsupported activation: {act}')


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            act='relu'):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab

        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False))
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False)

        self.bn = nn.BatchNorm2d(out_chs)
        self.act = _get_activation(act) if self.use_act else nn.Identity()
        self.lab = LearnableAffineBlock() if (self.use_act and self.use_lab) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, groups=1, use_lab=False, act='relu'):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            groups=groups,
            use_act=False,
            use_lab=use_lab,
            act=act)
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
            act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    """Stem block from DEIM HGNetv2."""

    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act='relu'):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab, act=act)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab, act=act)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab, act=act)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab, act=act)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act=act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return identity * x


class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
            act='relu'):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                        act=act))
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                        act=act))

        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act)
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act)
            self.aggregation = nn.Sequential(aggregation_squeeze_conv, aggregation_excitation_conv)
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act)
            self.aggregation = nn.Sequential(aggregation_conv, EseModule(out_chs))

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
            act='relu'):
        super().__init__()
        self.downsample = ConvBNAct(
            in_chs,
            in_chs,
            kernel_size=3,
            stride=2,
            groups=in_chs,
            use_act=False,
            use_lab=use_lab,
            act=act) if downsample else nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    act=act))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class DEIMStem(StemBlock):
    """YAML-facing DEIM HGNetv2 stem module."""

    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act='relu'):
        super().__init__(in_chs, mid_chs, out_chs, use_lab=use_lab, act=act)


class DEIMHGStage(HG_Stage):
    """YAML-facing DEIM HGNetv2 stage module."""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            act='relu'):
        super().__init__(
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=downsample,
            light_block=light_block,
            kernel_size=kernel_size,
            use_lab=use_lab,
            act=act)


__all__ = (
    'LearnableAffineBlock',
    'ConvBNAct',
    'LightConvBNAct',
    'StemBlock',
    'EseModule',
    'HG_Block',
    'HG_Stage',
    'DEIMStem',
    'DEIMHGStage',
)
