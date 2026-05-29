# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.ops import non_max_suppression
import torchvision
from cv2.gapi import kernel

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
    'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ResNetLayer', 'IN', 'Multiin', 'MF',
    'EntropyPrior', 'EntropyOffsetGateFusion', 'EntropyOffsetGateFusionLite', 'EntropyOffsetGateFusionEfficient')

import torch
from torch.nn import init, Sequential
import torch.nn as nn
import math
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
import torch.nn.functional as F
import numpy as np


def _normalize_spatial_map(x, eps=1e-6):
    x = x - x.amin(dim=(2, 3), keepdim=True)
    return x / (x.amax(dim=(2, 3), keepdim=True) + eps)


def _soft_histogram_local_shannon_entropy(x, kernel_size=5, num_bins=8, tau=0.125, eps=1e-6):
    """Differentiable soft-histogram local Shannon entropy approximation on scalar maps."""
    x = _normalize_spatial_map(x, eps=eps)
    centers = (torch.arange(num_bins, device=x.device, dtype=x.dtype) + 0.5) / num_bins
    diff = x.unsqueeze(2) - centers.view(1, 1, num_bins, 1, 1)
    logits = -diff.square() / max(tau, eps)
    assignments = torch.softmax(logits, dim=2).squeeze(1)
    local_probs = F.avg_pool2d(assignments, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    local_probs = local_probs / (local_probs.sum(dim=1, keepdim=True) + eps)
    entropy = -(local_probs * torch.log(local_probs + eps)).sum(dim=1, keepdim=True)
    return entropy / math.log(num_bins)


def _fast_local_entropy_proxy(x, kernel_size=5, eps=1e-6):
    """Fast local entropy proxy based on normalized local variance."""
    x = _normalize_spatial_map(x, eps=eps)
    mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    mean_sq = F.avg_pool2d(x.square(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    var = (mean_sq - mean.square()).clamp_min(0.0)
    return _normalize_spatial_map(torch.log1p(var * 16.0), eps=eps)


class EntropyPrior(nn.Module):
    def __init__(self, entropy_ks=5, hidden_dim=16, downsample=1, fast=False):
        super().__init__()
        self.entropy_ks = entropy_ks
        self.downsample = max(int(downsample), 1)
        self.fast = bool(fast)
        self.entropy_bins = 8
        self.entropy_tau = 0.125
        self.entropy_eps = 1e-6
        self.head = nn.Sequential(
            nn.Conv2d(5, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    @staticmethod
    def _normalize_map(x, eps=1e-6):
        return _normalize_spatial_map(x, eps=eps)

    def _local_spatial_entropy(self, x, eps=1e-6):
        if self.fast:
            return _fast_local_entropy_proxy(x, kernel_size=self.entropy_ks, eps=max(eps, self.entropy_eps))
        return _soft_histogram_local_shannon_entropy(
            x,
            kernel_size=self.entropy_ks,
            num_bins=self.entropy_bins,
            tau=self.entropy_tau,
            eps=max(eps, self.entropy_eps),
        )

    def forward(self, x):
        if self.downsample > 1:
            x = F.avg_pool2d(x, kernel_size=self.downsample, stride=self.downsample)

        split = x.shape[1] // 2
        x_rgb = x[:, :split, :, :]
        x_ir = x[:, split:, :, :]

        rgb_intensity = x_rgb.mean(dim=1, keepdim=True)
        ir_intensity = x_ir.mean(dim=1, keepdim=True)
        rgb_norm = self._normalize_map(rgb_intensity)
        ir_norm = self._normalize_map(ir_intensity)
        rgb_entropy = self._local_spatial_entropy(rgb_intensity)
        ir_entropy = self._local_spatial_entropy(ir_intensity)

        cues = torch.cat((rgb_norm, ir_norm, rgb_entropy, ir_entropy, torch.abs(rgb_norm - ir_norm)), dim=1)
        return self.head(cues)


class EntropyOffsetGateFusion(nn.Module):
    def __init__(self, layer, beta=0.4, shared_dim=64, reduction=16, entropy_ks=5):
        super().__init__()
        self.layer = layer
        self.beta = beta
        self.shared_dim = shared_dim
        self.reduction = reduction
        self.entropy_ks = entropy_ks
        self.entropy_bins = 8
        self.entropy_tau = 0.125
        self.entropy_eps = 1e-6
        self.max_offset = 2.0

        self.rgb_proj = None
        self.ir_proj = None
        self.flow_head = None
        self.spatial_gate = None
        self.rgb_channel_gate = None
        self.ir_channel_gate = None
        self.kd_cache = {}
        self._grid_cache = {}

    def _build_layers(self, x_rgb):
        channels = x_rgb.shape[1]
        shared_hidden = max(self.shared_dim, 16)
        gate_hidden = max(self.shared_dim // max(self.reduction, 1), 8)

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(channels, self.shared_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.shared_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_proj = nn.Sequential(
            nn.Conv2d(channels, self.shared_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.shared_dim),
            nn.SiLU(inplace=True),
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(self.shared_dim * 3 + 1, shared_hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(shared_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(shared_hidden, 3, kernel_size=1, stride=1, padding=0),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.shared_dim * 2 + 4, shared_hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(shared_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(shared_hidden, 2, kernel_size=1, stride=1, padding=0),
        )
        self.rgb_channel_gate = nn.Sequential(
            nn.Linear(self.shared_dim * 2, gate_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, channels, bias=True),
            nn.Sigmoid(),
        )
        self.ir_channel_gate = nn.Sequential(
            nn.Linear(self.shared_dim * 2, gate_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, channels, bias=True),
            nn.Sigmoid(),
        )

        self.rgb_proj = self.rgb_proj.to(device=x_rgb.device)
        self.ir_proj = self.ir_proj.to(device=x_rgb.device)
        self.flow_head = self.flow_head.to(device=x_rgb.device)
        self.spatial_gate = self.spatial_gate.to(device=x_rgb.device)
        self.rgb_channel_gate = self.rgb_channel_gate.to(device=x_rgb.device)
        self.ir_channel_gate = self.ir_channel_gate.to(device=x_rgb.device)

    @staticmethod
    def _normalize_map(x, eps=1e-6):
        return _normalize_spatial_map(x, eps=eps)

    def _local_spatial_entropy(self, x, eps=1e-6):
        energy = self._normalize_map(x.abs().mean(dim=1, keepdim=True))
        return _soft_histogram_local_shannon_entropy(
            energy,
            kernel_size=self.entropy_ks,
            num_bins=self.entropy_bins,
            tau=self.entropy_tau,
            eps=max(eps, self.entropy_eps),
        )

    @staticmethod
    def _channel_entropy(x, eps=1e-6):
        probs = x.abs().flatten(2)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
        return entropy / math.log(max(probs.shape[-1], 2))

    @staticmethod
    def _make_grid(h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        if TORCH_1_10:
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        else:
            yy, xx = torch.meshgrid(ys, xs)
        return torch.stack((xx, yy), dim=-1).unsqueeze(0)

    def _warp(self, x, flow):
        h, w = x.shape[2:]
        key = (h, w, x.device.type, x.device.index, x.dtype)
        base_grid = self._grid_cache.get(key)
        if base_grid is None or base_grid.device != x.device or base_grid.dtype != x.dtype:
            base_grid = self._make_grid(h, w, x.device, x.dtype)
            self._grid_cache[key] = base_grid
        base_grid = base_grid.expand(x.shape[0], -1, -1, -1)
        flow_x = flow[:, 0] * (2.0 / max(w - 1, 1))
        flow_y = flow[:, 1] * (2.0 / max(h - 1, 1))
        sampling_grid = base_grid + torch.stack((flow_x, flow_y), dim=-1)
        return F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def _channel_gate(self, gate, feature):
        gap = F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)
        channel_entropy = self._channel_entropy(feature)
        gate_input = torch.cat((gap, 1.0 - channel_entropy), dim=1)
        return gate(gate_input).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        x_ir, x_rgb, prior = x
        if self.rgb_proj is None:
            self._build_layers(x_rgb)

        prior = F.interpolate(prior, size=x_ir.shape[2:], mode='bilinear', align_corners=False)
        z_rgb = self.rgb_proj(x_rgb)
        z_ir = self.ir_proj(x_ir)

        flow_input = torch.cat((z_rgb, z_ir, torch.abs(z_rgb - z_ir), prior), dim=1)
        flow_and_conf = self.flow_head(flow_input)
        flow = torch.tanh(flow_and_conf[:, :2]) * self.max_offset
        align_conf = torch.sigmoid(flow_and_conf[:, 2:3])

        z_rgb_aligned = self._warp(z_rgb, flow)
        x_rgb_aligned = self._warp(x_rgb, flow)

        hs_rgb = self._local_spatial_entropy(z_rgb_aligned)
        hs_ir = self._local_spatial_entropy(z_ir)
        spatial_input = torch.cat((z_rgb_aligned, z_ir, hs_rgb, hs_ir, prior, align_conf), dim=1)
        spatial_logits = self.spatial_gate(spatial_input)
        weights = torch.softmax(spatial_logits, dim=1)
        gate_entropy = -(
            weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()
        ).sum(dim=1, keepdim=True) / math.log(max(weights.shape[1], 2))

        w_rgb = 0.5 + self.beta * (weights[:, 0:1] - 0.5)
        w_ir = 1.0 - w_rgb
        c_rgb = self._channel_gate(self.rgb_channel_gate, z_rgb_aligned)
        c_ir = self._channel_gate(self.ir_channel_gate, z_ir)
        self.kd_cache = {
            "z_rgb_aligned": z_rgb_aligned,
            "z_ir": z_ir,
            "spatial_logits": spatial_logits,
            "gate_probs": weights,
            "gate_entropy": gate_entropy,
            "w_rgb": w_rgb,
            "w_ir": w_ir,
            "align_conf": align_conf,
            "c_rgb": c_rgb,
            "c_ir": c_ir,
        }
        return w_rgb * c_rgb * x_rgb_aligned + w_ir * c_ir * x_ir


class EntropyOffsetGateFusionEfficient(EntropyOffsetGateFusion):
    """Latency-oriented SEMD fusion: keep entropy gates, align only selected scales."""

    def __init__(
        self, layer, beta=0.4, shared_dim=48, reduction=16, entropy_ks=5, align=True, entropy_pool=1,
        warp_feature=True
    ):
        super().__init__(layer, beta=beta, shared_dim=shared_dim, reduction=reduction, entropy_ks=entropy_ks)
        self.align = bool(align)
        self.entropy_pool = max(int(entropy_pool), 1)
        self.warp_feature = bool(warp_feature)

    def _build_layers(self, x_rgb):
        channels = x_rgb.shape[1]
        shared_hidden = max(self.shared_dim, 16)
        gate_hidden = max(self.shared_dim // max(self.reduction, 1), 8)

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(channels, self.shared_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.shared_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_proj = nn.Sequential(
            nn.Conv2d(channels, self.shared_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.shared_dim),
            nn.SiLU(inplace=True),
        )
        if self.align:
            self.flow_head = nn.Sequential(
                nn.Conv2d(self.shared_dim * 3 + 1, shared_hidden, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(shared_hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(shared_hidden, 3, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.flow_head = None
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.shared_dim * 2 + 4, shared_hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(shared_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(shared_hidden, 2, kernel_size=1, stride=1, padding=0),
        )
        self.rgb_channel_gate = nn.Sequential(
            nn.Linear(self.shared_dim * 2, gate_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, channels, bias=True),
            nn.Sigmoid(),
        )
        self.ir_channel_gate = nn.Sequential(
            nn.Linear(self.shared_dim * 2, gate_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, channels, bias=True),
            nn.Sigmoid(),
        )

        self.rgb_proj = self.rgb_proj.to(device=x_rgb.device)
        self.ir_proj = self.ir_proj.to(device=x_rgb.device)
        if self.flow_head is not None:
            self.flow_head = self.flow_head.to(device=x_rgb.device)
        self.spatial_gate = self.spatial_gate.to(device=x_rgb.device)
        self.rgb_channel_gate = self.rgb_channel_gate.to(device=x_rgb.device)
        self.ir_channel_gate = self.ir_channel_gate.to(device=x_rgb.device)

    def _efficient_spatial_entropy(self, x):
        spatial_size = x.shape[2:]
        if self.entropy_pool <= 1 or min(x.shape[2:]) < self.entropy_pool:
            pooled = x
        else:
            pooled = F.avg_pool2d(x, kernel_size=self.entropy_pool, stride=self.entropy_pool)
        energy = self._normalize_map(pooled.abs().mean(dim=1, keepdim=True))
        entropy = _fast_local_entropy_proxy(
            energy, kernel_size=self.entropy_ks, eps=max(1e-6, self.entropy_eps)
        )
        if entropy.shape[2:] != spatial_size:
            entropy = F.interpolate(entropy, size=spatial_size, mode='bilinear', align_corners=False)
        return entropy

    def forward(self, x):
        x_ir, x_rgb, prior = x
        if self.rgb_proj is None:
            self._build_layers(x_rgb)

        prior = F.interpolate(prior, size=x_ir.shape[2:], mode='bilinear', align_corners=False)
        z_rgb = self.rgb_proj(x_rgb)
        z_ir = self.ir_proj(x_ir)

        if self.align:
            flow_input = torch.cat((z_rgb, z_ir, torch.abs(z_rgb - z_ir), prior), dim=1)
            flow_and_conf = self.flow_head(flow_input)
            flow = torch.tanh(flow_and_conf[:, :2]) * self.max_offset
            align_conf = torch.sigmoid(flow_and_conf[:, 2:3])
            z_rgb_aligned = self._warp(z_rgb, flow)
            x_rgb_aligned = self._warp(x_rgb, flow) if self.warp_feature else x_rgb
        else:
            align_conf = torch.ones(
                x_ir.shape[0], 1, x_ir.shape[2], x_ir.shape[3], device=x_ir.device, dtype=x_ir.dtype
            )
            z_rgb_aligned = z_rgb
            x_rgb_aligned = x_rgb

        hs_rgb = self._efficient_spatial_entropy(z_rgb_aligned)
        hs_ir = self._efficient_spatial_entropy(z_ir)
        spatial_input = torch.cat((z_rgb_aligned, z_ir, hs_rgb, hs_ir, prior, align_conf), dim=1)
        spatial_logits = self.spatial_gate(spatial_input)
        weights = torch.softmax(spatial_logits, dim=1)
        gate_entropy = -(
            weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()
        ).sum(dim=1, keepdim=True) / math.log(max(weights.shape[1], 2))

        w_rgb = 0.5 + self.beta * (weights[:, 0:1] - 0.5)
        w_ir = 1.0 - w_rgb
        c_rgb = self._channel_gate(self.rgb_channel_gate, z_rgb_aligned)
        c_ir = self._channel_gate(self.ir_channel_gate, z_ir)
        self.kd_cache = {
            "z_rgb_aligned": z_rgb_aligned,
            "z_ir": z_ir,
            "spatial_logits": spatial_logits,
            "gate_probs": weights,
            "gate_entropy": gate_entropy,
            "w_rgb": w_rgb,
            "w_ir": w_ir,
            "align_conf": align_conf,
            "c_rgb": c_rgb,
            "c_ir": c_ir,
        }
        return w_rgb * c_rgb * x_rgb_aligned + w_ir * c_ir * x_ir


class EntropyOffsetGateFusionLite(EntropyOffsetGateFusion):
    """Lite E2 fusion: keep SEMD gates/alignment, lighten 3x3 heads with DW+PW convs."""

    def __init__(self, layer, beta=0.4, shared_dim=32, reduction=16, entropy_ks=5, align=True):
        super().__init__(layer, beta=beta, shared_dim=shared_dim, reduction=reduction, entropy_ks=entropy_ks)
        self.align = align

    @staticmethod
    def _lite_head(in_channels, hidden_channels, out_channels):
        return nn.Sequential(
            DWConv(in_channels, in_channels, k=3, s=1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def _build_layers(self, x_rgb):
        channels = x_rgb.shape[1]
        shared_hidden = max(self.shared_dim, 16)
        gate_hidden = max(self.shared_dim // max(self.reduction, 1), 8)

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(channels, self.shared_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.shared_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_proj = nn.Sequential(
            nn.Conv2d(channels, self.shared_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.shared_dim),
            nn.SiLU(inplace=True),
        )
        self.flow_head = self._lite_head(self.shared_dim * 3 + 1, shared_hidden, 3)
        self.spatial_gate = self._lite_head(self.shared_dim * 2 + 4, shared_hidden, 2)
        self.rgb_channel_gate = nn.Sequential(
            nn.Linear(self.shared_dim * 2, gate_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, channels, bias=True),
            nn.Sigmoid(),
        )
        self.ir_channel_gate = nn.Sequential(
            nn.Linear(self.shared_dim * 2, gate_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, channels, bias=True),
            nn.Sigmoid(),
        )

        self.rgb_proj = self.rgb_proj.to(device=x_rgb.device)
        self.ir_proj = self.ir_proj.to(device=x_rgb.device)
        self.flow_head = self.flow_head.to(device=x_rgb.device)
        self.spatial_gate = self.spatial_gate.to(device=x_rgb.device)
        self.rgb_channel_gate = self.rgb_channel_gate.to(device=x_rgb.device)
        self.ir_channel_gate = self.ir_channel_gate.to(device=x_rgb.device)

    def forward(self, x):
        x_ir, x_rgb, prior = x
        if self.rgb_proj is None:
            self._build_layers(x_rgb)

        prior = F.interpolate(prior, size=x_ir.shape[2:], mode='bilinear', align_corners=False)
        z_rgb = self.rgb_proj(x_rgb)
        z_ir = self.ir_proj(x_ir)

        flow_input = torch.cat((z_rgb, z_ir, torch.abs(z_rgb - z_ir), prior), dim=1)
        flow_and_conf = self.flow_head(flow_input)
        flow = torch.tanh(flow_and_conf[:, :2]) * self.max_offset
        align_conf = torch.sigmoid(flow_and_conf[:, 2:3])

        if self.align:
            z_rgb_aligned = self._warp(z_rgb, flow)
            x_rgb_aligned = self._warp(x_rgb, flow)
        else:
            z_rgb_aligned = z_rgb
            x_rgb_aligned = x_rgb
            align_conf = torch.ones_like(align_conf)

        hs_rgb = self._local_spatial_entropy(z_rgb_aligned)
        hs_ir = self._local_spatial_entropy(z_ir)
        spatial_input = torch.cat((z_rgb_aligned, z_ir, hs_rgb, hs_ir, prior, align_conf), dim=1)
        spatial_logits = self.spatial_gate(spatial_input)
        weights = torch.softmax(spatial_logits, dim=1)
        gate_entropy = -(
            weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()
        ).sum(dim=1, keepdim=True) / math.log(max(weights.shape[1], 2))

        w_rgb = 0.5 + self.beta * (weights[:, 0:1] - 0.5)
        w_ir = 1.0 - w_rgb
        c_rgb = self._channel_gate(self.rgb_channel_gate, z_rgb_aligned)
        c_ir = self._channel_gate(self.ir_channel_gate, z_ir)
        self.kd_cache = {
            "z_rgb_aligned": z_rgb_aligned,
            "z_ir": z_ir,
            "spatial_logits": spatial_logits,
            "gate_probs": weights,
            "gate_entropy": gate_entropy,
            "w_rgb": w_rgb,
            "w_ir": w_ir,
            "align_conf": align_conf,
            "c_rgb": c_rgb,
            "c_ir": c_ir,
        }
        return w_rgb * c_rgb * x_rgb_aligned + w_ir * c_ir * x_ir


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MF(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(MF, self).__init__()
        self.mask_map_r = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):
        x_left_ori, x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out


class IN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Multiin(nn.Module):  # stereo attention block
    def __init__(self, out=1):
        super().__init__()
        self.out = out

    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        if self.out == 1:
            x = x1
        else:
            x = x2
        return x


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(Conv(c1, c2, k=7, s=2, p=3, act=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)

# class LinearProbe(nn.Module):
#     """YOLOv8 LinearProbe head for detection models."""
#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init
#
#     def __init__(self, nc=80, ch=()):
#         """Initializes the YOLOv8 LinearProbe layer with specified number of classes and channels."""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#
#         # cv2 -> 锚框, cv3 -> 类别
#         self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
#         self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
#
#     def forward(self, x):
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         if len(x) > 3:
#             x = x[3:]
#
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:  # Training path
#             return x
#
#         # Inference path
#         shape = x[0].shape  # BCHW
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape
#
#         if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
#             box = x_cat[:, :self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4:]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
#         dbox = self.decode_bboxes(box)
#
#         if self.export and self.format in ('tflite', 'edgetpu'):
#             # Precompute normalization factor to increase numerical stability
#             # See https://github.com/ultralytics/ultralytics/issues/7371
#             img_h = shape[2]
#             img_w = shape[3]
#             img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
#             norm = self.strides / (self.stride[0] * img_size)
#             dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)
#
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#
#         return y if self.export else (y, x)
#
#     def bias_init(self):
#         """Initialize LinearProbe() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # LinearProbe() module
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a.bias.data[:] = 1.0  # box
#             b.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
#
#     def decode_bboxes(self, bboxes):
#         """Decode bounding boxes."""
#
#         return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
