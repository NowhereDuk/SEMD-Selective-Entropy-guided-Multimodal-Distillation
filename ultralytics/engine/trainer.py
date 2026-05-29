# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco128.yaml imgsz=640 epochs=100 batch=16
"""

import ast
import csv
import json
import math
import os
import subprocess
import sys
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
import torch.nn.functional as F

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, TQDM, __version__, callbacks, clean_url, colorstr, emojis,
                               yaml_save)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle, select_device,
                                           strip_optimizer)

from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox, RotatedTaskAlignedAssigner
from ultralytics.utils.loss import RotatedBboxLoss
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


def normalize_module_hook_dicts(model):
    """Convert deserialized hook dicts back to weakref-able OrderedDicts before registering hooks."""
    if model is None:
        return
    hook_attrs = (
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    for module in de_parallel(model).modules():
        for attr in hook_attrs:
            value = getattr(module, attr, None)
            if isinstance(value, dict) and not isinstance(value, OrderedDict):
                setattr(module, attr, OrderedDict(value))


class TrainingHeadroomAuditLogger:
    """Lightweight epoch-level logger for training headroom audits."""

    def __init__(self, save_dir, output_name='headroom_audit'):
        self.audit_dir = Path(save_dir) / output_name
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.audit_dir / 'audit_metrics.csv'
        self.json_path = self.audit_dir / 'audit_metrics.json'
        self.jsonl_path = self.audit_dir / 'audit_metrics.jsonl'
        self.checkpoints_dir = self.audit_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_index_path = self.audit_dir / 'checkpoint_index.json'
        self.probe_manifest_path = self.audit_dir / 'probe_manifest.json'
        self.probe_results_csv = self.audit_dir / 'probe_results.csv'
        self.probe_results_json = self.audit_dir / 'probe_results.json'

    @staticmethod
    def _json_safe(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return TrainingHeadroomAuditLogger._json_safe(value.item())
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: TrainingHeadroomAuditLogger._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrainingHeadroomAuditLogger._json_safe(v) for v in value]
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        return value

    @classmethod
    def _csv_safe(cls, value):
        value = cls._json_safe(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if value is None:
            return ''
        return value

    def _append_csv_row(self, path, record):
        record = {k: self._csv_safe(v) for k, v in record.items()}
        if path.exists():
            with path.open('r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
        else:
            rows = []
            fieldnames = []

        new_keys = [k for k in record.keys() if k not in fieldnames]
        if new_keys:
            fieldnames = fieldnames + new_keys if fieldnames else list(record.keys())
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                writer.writerow(record)
        else:
            with path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not fieldnames:
                    writer.writeheader()
                writer.writerow(record)

    def append_epoch_record(self, record):
        safe_record = {k: self._json_safe(v) for k, v in record.items()}
        self._append_csv_row(self.csv_path, safe_record)
        if self.json_path.exists():
            with self.json_path.open('r', encoding='utf-8') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        else:
            history = []
        history.append(safe_record)
        with self.json_path.open('w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        with self.jsonl_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + '\n')

    def write_checkpoint_index(self, records):
        with self.checkpoint_index_path.open('w', encoding='utf-8') as f:
            json.dump(self._json_safe(records), f, ensure_ascii=False, indent=2)

    def write_probe_manifest(self, manifest):
        with self.probe_manifest_path.open('w', encoding='utf-8') as f:
            json.dump(self._json_safe(manifest), f, ensure_ascii=False, indent=2)

    def write_probe_results(self, results):
        safe_results = [self._json_safe(row) for row in results]
        with self.probe_results_json.open('w', encoding='utf-8') as f:
            json.dump(safe_results, f, ensure_ascii=False, indent=2)
        if self.probe_results_csv.exists():
            self.probe_results_csv.unlink()
        for row in safe_results:
            self._append_csv_row(self.probe_results_csv, row)


class AuditProbeRegistry:
    """Registry for short-run audit probe actions."""

    DEFAULT_PROBES = ('continue', 'kd_off', 'normal_only', 'cross_off', 'lr_x0.3')

    def __init__(self):
        self._actions = {}
        self.register('continue', lambda _ctx: {}, description='Continue with the original configuration.')
        self.register(
            'kd_off',
            lambda _ctx: {
                'distill_disable_all': True,
                'distill_weight': 0.0,
                'distill_cross_attention': False,
                'distill_normal_distillation': False,
            },
            description='Disable all KD paths from the checkpoint onward.',
        )
        self.register(
            'normal_only',
            lambda _ctx: {
                'distill_cross_attention': False,
                'distill_normal_distillation': True,
            },
            description='Keep normal distillation but disable cross-attention distillation.',
        )
        self.register(
            'normal_only_gate_off',
            lambda _ctx: {
                'distill_cross_attention': False,
                'distill_normal_distillation': True,
                'distill_disable_gate_kd': True,
            },
            description='Keep normal distillation, disable cross-attention, and disable gate KD.',
        )
        self.register(
            'cross_off',
            lambda _ctx: {'distill_cross_attention': False},
            description='Disable cross-attention distillation and keep the rest unchanged.',
        )
        self.register(
            'lr_x0.3',
            lambda _ctx: {'lr_scale': 0.3},
            description='Resume with learning rate multiplied by 0.3.',
        )
        self.register(
            'head_cls_only',
            lambda ctx: {
                'distill_stage2_cls_only': True,
                'distill_resume_ckpt': ctx['checkpoint_path'],
                'freeze_backbone_fusion': True,
                'train_neck_head_only': True,
                'stage2_disable_early_stop': True,
                'use_probe_epochs_as_stage2_epochs': True,
            },
            available=lambda ctx: bool(ctx.get('supports_head_cls_only')),
            description='Switch to late-stage head cls-only KD from the checkpoint.',
        )
        self.register(
            'gate_kd_off',
            lambda _ctx: {'distill_disable_gate_kd': True},
            available=lambda ctx: bool(ctx.get('supports_gate_kd_toggle')),
            description='Disable gate KD while keeping the rest of the distillation path unchanged.',
        )
        self.register(
            'pseudo_fusion_off',
            lambda _ctx: {'distill_disable_pseudo_fusion_kd': True},
            available=lambda ctx: bool(ctx.get('supports_pseudo_fusion_kd_toggle')),
            description='Disable pseudo-fusion KD while keeping the rest of the distillation path unchanged.',
        )
        self.register(
            'strict_only_normal',
            lambda _ctx: {
                'distill_cross_attention': False,
                'distill_normal_distillation': True,
                'distill_disable_gate_kd': True,
                'distill_disable_pseudo_fusion_kd': True,
            },
            description='Keep normal branch KD while disabling cross, gate, and pseudo-fusion KD.',
        )
        self.register(
            'strict_plus_gate_norm_0.03_conf025',
            lambda _ctx: {
                'distill_cross_attention': False,
                'distill_normal_distillation': True,
                'distill_disable_gate_kd': False,
                'distill_disable_pseudo_fusion_kd': True,
                'distill_gate_kd_mode': 'normalized',
                'distill_gate_kd_weight': 0.03,
                'distill_gate_kd_mask_mode': 'conf_binary_soft',
                'distill_gate_kd_conf_thr': 0.25,
            },
            description='Keep strict normal KD while re-enabling normalized gate KD at weight 0.03.',
        )
        self.register(
            'strict_head_cls_only',
            lambda _ctx: {
                'distill_cross_attention': False,
                'distill_normal_distillation': True,
                'distill_disable_gate_kd': True,
                'distill_disable_pseudo_fusion_kd': True,
                'distill_head_kd_policy': 'cls_only',
            },
            description='Keep strict normal KD with only head cls KD enabled.',
        )
        self.register(
            'strict_head_kd_off',
            lambda _ctx: {
                'distill_cross_attention': False,
                'distill_normal_distillation': True,
                'distill_disable_gate_kd': True,
                'distill_disable_pseudo_fusion_kd': True,
                'distill_head_kd_policy': 'off',
            },
            description='Keep strict normal KD with all head KD disabled.',
        )
        self.register(
            'strict_head_geom_only',
            lambda _ctx: {
                'distill_cross_attention': False,
                'distill_normal_distillation': True,
                'distill_disable_gate_kd': True,
                'distill_disable_pseudo_fusion_kd': True,
                'distill_head_kd_policy': 'geom_only',
            },
            description='Keep strict normal KD with only head geometry KD enabled.',
        )
        self.register(
            'gate_norm_0.01',
            lambda _ctx: {
                'distill_gate_kd_mode': 'normalized',
                'distill_gate_kd_weight': 0.01,
                'distill_gate_kd_mask_mode': 'none',
            },
            description='Use normalized gate KD with weight 0.01 and no reliability mask.',
        )
        self.register(
            'gate_norm_0.03',
            lambda _ctx: {
                'distill_gate_kd_mode': 'normalized',
                'distill_gate_kd_weight': 0.03,
                'distill_gate_kd_mask_mode': 'none',
            },
            description='Use normalized gate KD with weight 0.03 and no reliability mask.',
        )
        self.register(
            'gate_norm_0.03_conf025',
            lambda _ctx: {
                'distill_gate_kd_mode': 'normalized',
                'distill_gate_kd_weight': 0.03,
                'distill_gate_kd_mask_mode': 'conf_binary_soft',
                'distill_gate_kd_conf_thr': 0.25,
            },
            description='Use normalized gate KD with confidence-aware masking at threshold 0.25.',
        )
        self.register(
            'gate_norm_0.05_conf025',
            lambda _ctx: {
                'distill_gate_kd_mode': 'normalized',
                'distill_gate_kd_weight': 0.05,
                'distill_gate_kd_mask_mode': 'conf_binary_soft',
                'distill_gate_kd_conf_thr': 0.25,
            },
            description='Use normalized gate KD with stronger weight and confidence-aware masking.',
        )
        self.register(
            'late_gate_off_48',
            lambda _ctx: {
                'distill_late_kd_start_epoch': 48,
                'distill_late_kd_policy': 'gate_off',
            },
            description='Disable gate KD once training reaches epoch 48.',
        )
        self.register(
            'late_strict_normal_48',
            lambda _ctx: {
                'distill_late_kd_start_epoch': 48,
                'distill_late_kd_policy': 'strict_only_normal',
            },
            description='Switch to strict-only-normal KD once training reaches epoch 48.',
        )

    def register(self, name, builder, available=None, description=''):
        self._actions[name] = {
            'builder': builder,
            'available': available or (lambda _ctx: True),
            'description': description,
        }

    @staticmethod
    def parse_probe_list(raw):
        if raw is None:
            return list(AuditProbeRegistry.DEFAULT_PROBES)
        if isinstance(raw, str):
            return [item.strip() for item in raw.split(',') if item.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(item).strip() for item in raw if str(item).strip()]
        raise TypeError(f'Unsupported probe list type: {type(raw).__name__}')

    @staticmethod
    def parse_probe_fractions(raw):
        if raw is None:
            return []
        if isinstance(raw, str):
            values = [item.strip() for item in raw.split(',') if item.strip()]
        elif isinstance(raw, (list, tuple)):
            values = raw
        else:
            raise TypeError(f'Unsupported probe fraction type: {type(raw).__name__}')
        fractions = []
        for value in values:
            frac = float(value)
            if frac <= 0.0 or frac >= 1.0:
                raise ValueError(f'Probe fraction must be in (0, 1), got {value!r}.')
            fractions.append(frac)
        return sorted(set(fractions))

    def expand(self, context, requested=None):
        probes = []
        skipped = []
        seen = set()
        for name in self.parse_probe_list(requested):
            if name in seen:
                continue
            seen.add(name)
            action = self._actions.get(name)
            if action is None:
                skipped.append({'name': name, 'reason': 'unknown_probe'})
                continue
            if not action['available'](context):
                skipped.append({'name': name, 'reason': 'unsupported_in_current_repo'})
                continue
            probes.append({
                'name': name,
                'description': action['description'],
                'overrides': action['builder'](context) or {},
            })
        return probes, skipped


class CWDLoss(nn.Module):

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """

        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # normalize in channel diemension
            import torch.nn.functional as F
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss


class PKDLoss(nn.Module):
    def __init__(self, channels_s, channels_t):
        super(PKDLoss, self).__init__()

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """

        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            ################ 计算皮尔逊相关系数PCC ################
            s_flat = s.view(N, C, -1)
            t_flat = t.view(N, C, -1)

            mean_s = torch.mean(s_flat, dim=-1, keepdim=True)
            mean_t = torch.mean(t_flat, dim=-1, keepdim=True)

            sm = s_flat - mean_s
            tm = t_flat - mean_t

            pcc_num = torch.sum(sm * tm, dim=-1)
            # Clamp before sqrt so zero-variance channels do not create infinite gradients in backward.
            pcc_den = torch.sqrt((torch.sum(sm ** 2, dim=-1) * torch.sum(tm ** 2, dim=-1)).clamp_min(1e-12))
            pcc = pcc_num / pcc_den

            ################ 计算PKD损失 ################
            pkd_loss = 1 - torch.mean(pcc)
            losses.append(pkd_loss)

        loss = sum(losses)
        return loss


class CrossAttentionLoss(nn.Module):
    def __init__(self, channels_s, channels_t, loss_weight=1.0):
        super(CrossAttentionLoss, self).__init__()
        self.loss_weight = loss_weight

        # 将学生特征与老师特征对齐
        self.align_module = nn.ModuleList([
            nn.Conv2d(student_channel, teacher_channel, kernel_size=1, stride=1, padding=0)
            for student_channel, teacher_channel in zip(channels_s, channels_t)
        ])
        # 教师特征的BN层
        self.norm_t = nn.ModuleList([
            nn.BatchNorm2d(teacher_channel, affine=False)
            for teacher_channel in channels_t
        ])
        # 学生特征的BN层
        self.norm_s = nn.ModuleList([
            nn.BatchNorm2d(student_channel, affine=False)
            for student_channel in channels_s
        ])

        self.cross_attention_loss = CADLoss(channels_s, channels_t)

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        teacher_features = []
        student_features = []

        for i, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[i](s)
            s = self.norm_t[i](s)
            t = self.norm_t[i](t)

            teacher_features.append(t)
            student_features.append(s)

        loss = self.cross_attention_loss(student_features, teacher_features)
        return self.loss_weight * loss


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='cwd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller

        # 将学生特征与老师特征对齐
        self.align_module = nn.ModuleList([
            nn.Conv2d(student_channel, teacher_channel, kernel_size=1, stride=1, padding=0)
            for student_channel, teacher_channel in zip(channels_s, channels_t)
        ])
        # 教师特征的BN层
        self.norm_t = nn.ModuleList([
            nn.BatchNorm2d(teacher_channel, affine=False)
            for teacher_channel in channels_t
        ])
        # 学生特征的BN层
        self.norm_s = nn.ModuleList([
            nn.BatchNorm2d(student_channel, affine=False)
            for student_channel in channels_s
        ])

        # 选择蒸馏方法
        if distiller == 'CWD':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        elif distiller == 'PKD':
            self.feature_loss = PKDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):

        assert len(y_s) == len(y_t)
        teacher_features = []
        student_features = []

        for i, (s, t) in enumerate(zip(y_s, y_t)):

            if self.distiller == 'CWD' or self.distiller == 'PKD':
                s = self.align_module[i](s)
                s = self.norm_t[i](s)

            t = self.norm_t[i](t)
            teacher_features.append(t)
            student_features.append(s)

        loss = self.feature_loss(student_features, teacher_features)
        return self.loss_weight * loss


class CADLoss(nn.Module):
    def __init__(self, channels_s, channels_t, e_lambda=1e-4):
        super(CADLoss, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def SimAM(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)

    def attentionMAP(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activation(y)

    def forward(self, y_s, y_t):
        if len(y_t) == 6:  # 加入mrl后就会有奇怪的bug，还是得多学一下钩子
            y_t = y_t[:3]
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            attention_map = self.attentionMAP(t)
            s = s * attention_map
            t = t * attention_map
            N, C, H, W = s.shape

            ################ 计算皮尔逊相关系数PCC ################
            s_flat = s.view(N, C, -1)
            t_flat = t.view(N, C, -1)

            mean_s = torch.mean(s_flat, dim=-1, keepdim=True)
            mean_t = torch.mean(t_flat, dim=-1, keepdim=True)

            sm = s_flat - mean_s
            tm = t_flat - mean_t

            pcc_num = torch.sum(sm * tm, dim=-1)
            # Clamp before sqrt so zero-variance channels do not create infinite gradients in backward.
            pcc_den = torch.sqrt((torch.sum(sm ** 2, dim=-1) * torch.sum(tm ** 2, dim=-1)).clamp_min(1e-12))
            pcc = pcc_num / pcc_den

            ################ 计算PKD损失 ################
            pkd_loss = 1 - torch.mean(pcc)
            losses.append(pkd_loss)

        loss = sum(losses)

        return loss


class Distillation_loss(nn.Module):
    def __init__(self, student_model, teacher_model, distiller="CWD"):
        super(Distillation_loss, self).__init__()
        self.distiller = distiller
        normalize_module_hook_dicts(student_model)
        normalize_module_hook_dicts(teacher_model)

        layers_t = ["15", "18", "21"]  # 教师的用于计算蒸馏损失的层数
        layers_s = ["15", "18", "21"]  # 学生的用于计算蒸馏损失的层数
        length_t = len(layers_t)
        length_s = len(layers_s)
        assert length_t == length_s
        channels_s = [64, 128, 256]
        channels_t = [128, 256, 512]

        self.D_loss_fn = FeatureLoss(channels_s=channels_s, channels_t=channels_t, distiller=distiller)

        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []

        # 教师模型的特征提取层
        for mname, ml in teacher_model.named_modules():
            if mname is not None:
                name = mname.split(".")
                if name[0] == "module":
                    name.pop(0)
                if len(name) == 3:
                    if name[1] in layers_t:
                        if "cv2" in mname:
                            self.teacher_module_pairs.append(ml)

        # 学生模型的特征提取层
        for mname, ml in student_model.named_modules():
            if mname is not None:
                name = mname.split(".")
                if name[0] == "module":
                    name.pop(0)
                if len(name) == 3:
                    if name[1] in layers_s:
                        if "cv2" in mname:
                            self.student_module_pairs.append(ml)

    def register_hook(self):
        self.teacher_outputs = []
        self.student_outputs = []

        def make_layer_forward_hook(layer):
            def forward_hook(m, input, output):
                layer.append(output)

            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            # 为每层加入钩子，在进行Forward的时候会自动将每层的特征传送给model_outputs和student_outputs
            self.remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(self.student_outputs)))

    def get_loss(self):
        quant_loss = 0
        quant_loss += self.D_loss_fn(y_t=self.teacher_outputs, y_s=self.student_outputs)
        if self.distiller == 'MGD':
            quant_loss *= 0.3
        self.teacher_outputs.clear()
        self.student_outputs.clear()
        return quant_loss

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()


class StudentEntropyWeightHead(nn.Module):
    """Small bounded predictor for learnable student-entropy KD weighting."""

    def __init__(self, min_weight=0.5, max_weight=1.5, hidden_dim=8):
        super().__init__()
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self._init_output_bias()

    def _init_output_bias(self):
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        span = self.max_weight - self.min_weight
        if span <= 1e-6:
            nn.init.zeros_(final.bias)
            return
        target = (1.0 - self.min_weight) / span
        target = min(max(target, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(final.bias, math.log(target / (1.0 - target)))

    def forward(self, features):
        if self.max_weight <= self.min_weight:
            return features.new_full(features.shape[:-1], self.min_weight)
        logits = self.mlp(features).squeeze(-1)
        span = self.max_weight - self.min_weight
        return self.min_weight + span * torch.sigmoid(logits)


class Multimodal_Distillation_loss(nn.Module):
    """
    用于计算多模态模型的蒸馏损失。

    其中包含一个多模态学生模型、一个RGB单模态教师模型和一个IR单模态教师模型，
    两个教师分别蒸馏学生模型两路backbone特征。
    """

    GATE_KD_MODES = {'legacy', 'normalized'}
    GATE_KD_MASK_MODES = {'none', 'conf_binary', 'conf_soft', 'conf_binary_soft'}
    MID_KD_POLICY_IDS = {
        'none': 0,
        'normal_only_gate_off': 1,
    }
    LATE_KD_POLICY_IDS = {
        'none': 0,
        'gate_off': 1,
        'normal_gate_off': 2,
        'strict_only_normal': 3,
        'strict_head_kd_off': 4,
    }
    SCHEDULE_STAGE_IDS = {
        'base': 0,
        'mid': 1,
        'late': 2,
    }
    HEAD_KD_POLICY_IDS = {
        'full': 0,
        'cls_only': 1,
        'geom_only': 2,
        'off': 3,
    }

    def __init__(
            self,
            student_model,
            teacher_model_rgb,
            teacher_model_ir,
            distiller="PKD",
            cross_attention=True,
            normal_distillation=True,
            stage2_cls_only=False,
            distill_cls_kd_weight=0.05,
            distill_kd_temperature=2.0,
            teacher_conf_thr=0.45,
            teacher_entropy_thr=0.35,
            teacher_jsd_thr=0.10,
            distill_only_normal=False,
            distill_disable_gate_kd=False,
            distill_disable_pseudo_fusion_kd=False,
            distill_head_kd_policy='full',
            distill_schedule_enable=False,
            distill_mid_kd_start_epoch=-1,
            distill_mid_kd_policy='none',
            distill_gate_kd_mode='legacy',
            distill_gate_kd_weight=1.0,
            distill_gate_kd_temperature=1.0,
            distill_gate_kd_mask_mode='none',
            distill_gate_kd_conf_thr=0.25,
            distill_student_entropy_weight_enable=False,
            distill_student_entropy_weight_mode='fixed',
            distill_student_entropy_weight_formula='fixed_boost',
            distill_student_entropy_weight_min=0.5,
            distill_student_entropy_weight_max=1.5,
            distill_student_entropy_weight_beta=0.5,
            distill_student_entropy_weight_beta_pos=1.0,
            distill_student_entropy_weight_beta_neg=0.5,
            distill_student_entropy_weight_tau=0.5,
            distill_student_entropy_weight_normalize_mean=False,
            distill_student_entropy_weight_detach=True,
            distill_student_entropy_weight_target='gate',
            distill_student_entropy_weight_reg=0.0,
            distill_late_kd_start_epoch=-1,
            distill_late_kd_policy='none',
            student_rgb_layer_ids=None,
            student_ir_layer_ids=None,
            teacher_rgb_layer_ids=None,
            teacher_ir_layer_ids=None,
            student_rgb_channels=None,
            student_ir_channels=None,
            student_fusion_layer_ids=None,
            student_fusion_channels=None,
            teacher_rgb_channels=None,
            teacher_ir_channels=None):
        super(Multimodal_Distillation_loss, self).__init__()
        self.distiller = distiller
        self.stage2_cls_only = bool(stage2_cls_only)
        self.cross_attention = bool(cross_attention) and not self.stage2_cls_only
        self.normal_distillation = bool(normal_distillation) and not self.stage2_cls_only
        self.distill_cls_kd_weight = float(distill_cls_kd_weight)
        self.distill_kd_temperature = float(distill_kd_temperature)
        self.teacher_conf_thr = float(teacher_conf_thr)
        self.teacher_entropy_thr = float(teacher_entropy_thr)
        self.teacher_jsd_thr = float(teacher_jsd_thr)
        self.distill_only_normal = bool(distill_only_normal) and not self.stage2_cls_only
        self.distill_disable_gate_kd = bool(distill_disable_gate_kd)
        self.distill_disable_pseudo_fusion_kd = bool(distill_disable_pseudo_fusion_kd)
        self.distill_head_kd_policy = str(distill_head_kd_policy)
        self.distill_schedule_enable = bool(distill_schedule_enable)
        self.distill_mid_kd_start_epoch = int(distill_mid_kd_start_epoch)
        self.distill_mid_kd_policy = str(distill_mid_kd_policy)
        self.distill_gate_kd_mode = str(distill_gate_kd_mode)
        self.distill_gate_kd_weight = float(distill_gate_kd_weight)
        self.distill_gate_kd_temperature = float(distill_gate_kd_temperature)
        self.distill_gate_kd_mask_mode = str(distill_gate_kd_mask_mode)
        self.distill_gate_kd_conf_thr = float(distill_gate_kd_conf_thr)
        self.distill_student_entropy_weight_enable = bool(distill_student_entropy_weight_enable)
        self.distill_student_entropy_weight_mode = str(distill_student_entropy_weight_mode)
        self.distill_student_entropy_weight_formula = str(distill_student_entropy_weight_formula)
        self.distill_student_entropy_weight_min = float(distill_student_entropy_weight_min)
        self.distill_student_entropy_weight_max = float(distill_student_entropy_weight_max)
        self.distill_student_entropy_weight_beta = float(distill_student_entropy_weight_beta)
        self.distill_student_entropy_weight_beta_pos = float(distill_student_entropy_weight_beta_pos)
        self.distill_student_entropy_weight_beta_neg = float(distill_student_entropy_weight_beta_neg)
        self.distill_student_entropy_weight_tau = float(distill_student_entropy_weight_tau)
        self.distill_student_entropy_weight_normalize_mean = bool(distill_student_entropy_weight_normalize_mean)
        self.distill_student_entropy_weight_detach = bool(distill_student_entropy_weight_detach)
        self.distill_student_entropy_weight_target = str(distill_student_entropy_weight_target)
        self.distill_student_entropy_weight_reg = float(distill_student_entropy_weight_reg)
        self.distill_late_kd_start_epoch = int(distill_late_kd_start_epoch)
        self.distill_late_kd_policy = str(distill_late_kd_policy)
        if self.distill_gate_kd_mode not in self.GATE_KD_MODES:
            raise ValueError(f'Unsupported distill_gate_kd_mode: {self.distill_gate_kd_mode}')
        if self.distill_gate_kd_mask_mode not in self.GATE_KD_MASK_MODES:
            raise ValueError(f'Unsupported distill_gate_kd_mask_mode: {self.distill_gate_kd_mask_mode}')
        if self.distill_student_entropy_weight_mode not in {'fixed', 'learnable'}:
            raise ValueError(
                f'Unsupported distill_student_entropy_weight_mode: {self.distill_student_entropy_weight_mode}')
        if self.distill_student_entropy_weight_formula not in {'fixed_boost', 'centered', 'linear_map', 'asym_centered'}:
            raise ValueError(
                f'Unsupported distill_student_entropy_weight_formula: {self.distill_student_entropy_weight_formula}')
        if self.distill_student_entropy_weight_target not in {'gate', 'head_cls', 'gate_head_cls'}:
            raise ValueError(
                f'Unsupported distill_student_entropy_weight_target: {self.distill_student_entropy_weight_target}')
        if self.distill_head_kd_policy not in self.HEAD_KD_POLICY_IDS:
            raise ValueError(f'Unsupported distill_head_kd_policy: {self.distill_head_kd_policy}')
        if self.distill_mid_kd_policy not in self.MID_KD_POLICY_IDS:
            raise ValueError(f'Unsupported distill_mid_kd_policy: {self.distill_mid_kd_policy}')
        if self.distill_late_kd_policy not in self.LATE_KD_POLICY_IDS:
            raise ValueError(f'Unsupported distill_late_kd_policy: {self.distill_late_kd_policy}')

        self.student_model = self._unwrap_model(student_model)
        self.teacher_model_rgb = self._unwrap_model(teacher_model_rgb)
        self.teacher_model_ir = self._unwrap_model(teacher_model_ir)
        normalize_module_hook_dicts(self.student_model)
        normalize_module_hook_dicts(self.teacher_model_rgb)
        normalize_module_hook_dicts(self.teacher_model_ir)
        self.device = self._get_model_device(self.student_model)

        defaults = self._default_distill_config(self.student_model)

        self.student_rgb_layer_ids = self._normalize_int_list(
            student_rgb_layer_ids if student_rgb_layer_ids is not None else defaults['student_rgb_layer_ids'],
            'student_rgb_layer_ids')
        self.student_ir_layer_ids = self._normalize_int_list(
            student_ir_layer_ids if student_ir_layer_ids is not None else defaults['student_ir_layer_ids'],
            'student_ir_layer_ids')
        self.teacher_rgb_layer_ids = self._normalize_int_list(
            teacher_rgb_layer_ids if teacher_rgb_layer_ids is not None else defaults['teacher_rgb_layer_ids'],
            'teacher_rgb_layer_ids')
        self.teacher_ir_layer_ids = self._normalize_int_list(
            teacher_ir_layer_ids if teacher_ir_layer_ids is not None else defaults['teacher_ir_layer_ids'],
            'teacher_ir_layer_ids')

        self.student_rgb_channels = self._normalize_int_list(
            student_rgb_channels if student_rgb_channels is not None else defaults['student_rgb_channels'],
            'student_rgb_channels')
        self.student_ir_channels = self._normalize_int_list(
            student_ir_channels if student_ir_channels is not None else defaults['student_ir_channels'],
            'student_ir_channels')
        self.student_fusion_layer_ids = self._normalize_optional_int_list(
            student_fusion_layer_ids if student_fusion_layer_ids is not None else defaults['student_fusion_layer_ids'],
            'student_fusion_layer_ids')
        self.student_fusion_channels = self._normalize_optional_int_list(
            student_fusion_channels if student_fusion_channels is not None else defaults['student_fusion_channels'],
            'student_fusion_channels')
        self.teacher_rgb_channels = self._normalize_int_list(
            teacher_rgb_channels if teacher_rgb_channels is not None else defaults['teacher_rgb_channels'],
            'teacher_rgb_channels')
        self.teacher_ir_channels = self._normalize_int_list(
            teacher_ir_channels if teacher_ir_channels is not None else defaults['teacher_ir_channels'],
            'teacher_ir_channels')
        self.teacher_fusion_channels = (
            self.teacher_rgb_channels[1:1 + len(self.student_fusion_layer_ids)]
            if len(self.teacher_rgb_channels) >= len(self.student_fusion_layer_ids) + 1 else []
        )
        self.enable_finegrained = (
            len(self.student_fusion_layer_ids) > 0
            and len(self.teacher_rgb_layer_ids) == 4
            and len(self.teacher_ir_layer_ids) == 4
            and len(self.teacher_fusion_channels) == len(self.student_fusion_layer_ids)
        )

        self._validate_distill_config()

        if self.stage2_cls_only:
            self.D_loss_fn_rgb = None
            self.D_loss_fn_ir = None
            self.D_loss_fn_fusion = None
            self.Cross_loss_rgb_to_ir = None
            self.Cross_loss_ir_to_rgb = None
            self.teacher_module_pairs_rgb = []
            self.teacher_module_pairs_ir = []
            self.student_module_pairs_rgb = []
            self.student_module_pairs_ir = []
            self.student_module_pairs_fusion = []
        else:
            self.D_loss_fn_rgb = FeatureLoss(
                channels_s=self.student_rgb_channels,
                channels_t=self.teacher_rgb_channels,
                distiller=distiller)
            self.D_loss_fn_ir = FeatureLoss(
                channels_s=self.student_ir_channels,
                channels_t=self.teacher_ir_channels,
                distiller=distiller)
            self.D_loss_fn_fusion = (
                FeatureLoss(
                    channels_s=self.student_fusion_channels,
                    channels_t=self.teacher_fusion_channels,
                    distiller=distiller)
                if self.enable_finegrained else None
            )

            self.Cross_loss_rgb_to_ir = CrossAttentionLoss(
                channels_s=self.student_ir_channels,
                channels_t=self.teacher_rgb_channels)
            self.Cross_loss_ir_to_rgb = CrossAttentionLoss(
                channels_s=self.student_rgb_channels,
                channels_t=self.teacher_ir_channels)

            self.teacher_module_pairs_rgb = self._resolve_top_level_modules(
                self.teacher_model_rgb, self.teacher_rgb_layer_ids, 'teacher_rgb_layer_ids')
            self.teacher_module_pairs_ir = self._resolve_top_level_modules(
                self.teacher_model_ir, self.teacher_ir_layer_ids, 'teacher_ir_layer_ids')
            self.student_module_pairs_rgb = self._resolve_top_level_modules(
                self.student_model, self.student_rgb_layer_ids, 'student_rgb_layer_ids')
            self.student_module_pairs_ir = self._resolve_top_level_modules(
                self.student_model, self.student_ir_layer_ids, 'student_ir_layer_ids')
            self.student_module_pairs_fusion = self._resolve_top_level_modules(
                self.student_model, self.student_fusion_layer_ids, 'student_fusion_layer_ids'
            ) if self.student_fusion_layer_ids else []

        self.expected_feature_num = len(self.student_rgb_layer_ids)
        self.expected_fusion_num = len(self.student_fusion_layer_ids)
        self.remove_handle = []
        self.teacher_outputs_rgb = []
        self.teacher_outputs_ir = []
        self.student_outputs_rgb = []
        self.student_outputs_ir = []
        self.student_outputs_fusion = []
        self.last_loss_dict = {}
        self.student_entropy_weight_head = (
            StudentEntropyWeightHead(
                min_weight=self.distill_student_entropy_weight_min,
                max_weight=self.distill_student_entropy_weight_max,
            )
            if self.distill_student_entropy_weight_enable and self.distill_student_entropy_weight_mode == 'learnable'
            else None
        )

    @staticmethod
    def _unwrap_model(model):
        return de_parallel(model)

    @staticmethod
    def _get_model_device(model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    @staticmethod
    def _normalize_int_list(values, field_name):
        if isinstance(values, str):
            raw = values.strip()
            if not raw:
                raise ValueError(f'{field_name}: expected a non-empty list of integers.')
            try:
                values = ast.literal_eval(raw) if raw[0] in '[(' else [item.strip() for item in raw.split(',')]
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f'{field_name}: failed to parse integer list from {values!r}.') from exc

        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f'{field_name}: expected a list/tuple or list-like string, got {type(values).__name__}.')

        try:
            return [int(item) for item in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{field_name}: all entries must be integers, got {values!r}.') from exc

    @staticmethod
    def _normalize_optional_int_list(values, field_name):
        if values is None:
            return []
        if isinstance(values, str) and not values.strip():
            return []
        return Multimodal_Distillation_loss._normalize_int_list(values, field_name)

    @staticmethod
    def _default_distill_config(student_model):
        layer_types = [type(layer).__name__ for layer in student_model.model]

        if (
            len(layer_types) > 16
            and all(layer_types[idx] == 'DEIMHGStage' for idx in (6, 7, 8, 9, 11, 12, 14, 15))
            and all(layer_types[idx] == 'EntropyOffsetGateFusion' for idx in (10, 13, 16))
        ):
            return {
                'student_rgb_layer_ids': [6, 8, 11, 14],
                'student_ir_layer_ids': [7, 9, 12, 15],
                'student_fusion_layer_ids': [10, 13, 16],
                'teacher_rgb_layer_ids': [1, 2, 3, 4],
                'teacher_ir_layer_ids': [1, 2, 3, 4],
                'student_rgb_channels': [64, 256, 512, 1024],
                'student_ir_channels': [64, 256, 512, 1024],
                'student_fusion_channels': [256, 512, 1024],
                'teacher_rgb_channels': [64, 256, 512, 1024],
                'teacher_ir_channels': [64, 256, 512, 1024],
            }

        # Keep legacy tap defaults for older student configurations.
        if len(student_model.model) == 39:
            student_rgb_layer_ids = [12, 17, 22]
            student_ir_layer_ids = [13, 18, 23]
        else:
            student_rgb_layer_ids = [11, 16, 21]
            student_ir_layer_ids = [12, 17, 22]

        return {
            'student_rgb_layer_ids': student_rgb_layer_ids,
            'student_ir_layer_ids': student_ir_layer_ids,
            'student_fusion_layer_ids': [],
            'teacher_rgb_layer_ids': [4, 6, 8],
            'teacher_ir_layer_ids': [4, 6, 8],
            'student_rgb_channels': [192, 384, 576],
            'student_ir_channels': [192, 384, 576],
            'student_fusion_channels': [],
            'teacher_rgb_channels': [192, 384, 576],
            'teacher_ir_channels': [192, 384, 576],
        }

    def _resolve_top_level_modules(self, model, layer_ids, field_name):
        model = self._unwrap_model(model)
        if not hasattr(model, 'model'):
            raise AttributeError(f'{field_name}: target model has no top-level attribute "model".')

        top_modules = model.model
        total = len(top_modules)
        modules = []
        for idx in layer_ids:
            if idx < 0 or idx >= total:
                raise IndexError(
                    f'{field_name}: layer id {idx} out of range [0, {total - 1}] for model with {total} layers.')
            modules.append(top_modules[idx])
        return modules

    def _validate_distill_config(self):
        lens = {
            'student_rgb_layer_ids': len(self.student_rgb_layer_ids),
            'student_ir_layer_ids': len(self.student_ir_layer_ids),
            'teacher_rgb_layer_ids': len(self.teacher_rgb_layer_ids),
            'teacher_ir_layer_ids': len(self.teacher_ir_layer_ids),
            'student_rgb_channels': len(self.student_rgb_channels),
            'student_ir_channels': len(self.student_ir_channels),
            'teacher_rgb_channels': len(self.teacher_rgb_channels),
            'teacher_ir_channels': len(self.teacher_ir_channels),
        }
        expected = lens['student_rgb_layer_ids']
        for k, v in lens.items():
            if v != expected:
                raise ValueError(
                    f'Distillation config length mismatch: {k}={v}, expected {expected}. Full config lengths: {lens}.')
        if self.student_fusion_layer_ids and len(self.student_fusion_layer_ids) != len(self.student_fusion_channels):
            raise ValueError(
                'Distillation config length mismatch: '
                f'student_fusion_layer_ids={len(self.student_fusion_layer_ids)} vs '
                f'student_fusion_channels={len(self.student_fusion_channels)}.'
            )

    def register_hook(self):
        self.remove_handle_()
        self.teacher_outputs_rgb = []
        self.teacher_outputs_ir = []
        self.student_outputs_rgb = []
        self.student_outputs_ir = []
        self.student_outputs_fusion = []
        self.last_loss_dict = {}
        if self.stage2_cls_only:
            return

        def make_layer_forward_hook(layer):
            def forward_hook(_m, _input, output):
                layer.append(output)

            return forward_hook

        for module in self.teacher_module_pairs_rgb:
            self.remove_handle.append(module.register_forward_hook(make_layer_forward_hook(self.teacher_outputs_rgb)))
        for module in self.student_module_pairs_rgb:
            self.remove_handle.append(module.register_forward_hook(make_layer_forward_hook(self.student_outputs_rgb)))
        for module in self.teacher_module_pairs_ir:
            self.remove_handle.append(module.register_forward_hook(make_layer_forward_hook(self.teacher_outputs_ir)))
        for module in self.student_module_pairs_ir:
            self.remove_handle.append(module.register_forward_hook(make_layer_forward_hook(self.student_outputs_ir)))
        for module in self.student_module_pairs_fusion:
            self.remove_handle.append(module.register_forward_hook(make_layer_forward_hook(self.student_outputs_fusion)))

    def _validate_captured_features(self):
        counts = {
            'teacher_outputs_rgb': len(self.teacher_outputs_rgb),
            'teacher_outputs_ir': len(self.teacher_outputs_ir),
            'student_outputs_rgb': len(self.student_outputs_rgb),
            'student_outputs_ir': len(self.student_outputs_ir),
        }
        for k, v in counts.items():
            if v != self.expected_feature_num:
                raise RuntimeError(
                    f'Distillation hook capture mismatch: {k} has {v} features, expected {self.expected_feature_num}.')
        if self.expected_fusion_num and len(self.student_outputs_fusion) != self.expected_fusion_num:
            raise RuntimeError(
                'Distillation hook capture mismatch: '
                f'student_outputs_fusion has {len(self.student_outputs_fusion)} features, '
                f'expected {self.expected_fusion_num}.'
            )

    @staticmethod
    def _zero(device):
        return torch.zeros((), device=device)

    def _update_last_loss_dict(self, values):
        self.last_loss_dict = {
            k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
            for k, v in values.items()
        }

    def _weighted_masked_mean(self, loss, mask, weight=None):
        if loss.numel() == 0:
            return self._zero(self.device)
        if mask.ndim < loss.ndim:
            for _ in range(loss.ndim - mask.ndim):
                mask = mask.unsqueeze(-1)
        total_weight = mask.to(loss.dtype)
        if weight is not None:
            if weight.ndim < loss.ndim:
                for _ in range(loss.ndim - weight.ndim):
                    weight = weight.unsqueeze(-1)
            total_weight = total_weight * weight.to(loss.dtype)
        denom = total_weight.sum().clamp_min(1e-6)
        return (loss * total_weight).sum() / denom

    def _get_student_criterion_cache(self):
        criterion = getattr(self.student_model, 'criterion', None)
        return getattr(criterion, 'kd_cache', {}) if criterion is not None else {}

    @staticmethod
    def _tensor_mean_or_zero(value, device):
        if value is None:
            return torch.zeros((), device=device)
        return value.mean() if value.numel() else torch.zeros((), device=device)

    @staticmethod
    def _tensor_std_or_zero(value, device):
        if value is None or value.numel() <= 1:
            return torch.zeros((), device=device)
        return value.std(unbiased=False)

    @staticmethod
    def _select_valid_values(value, valid_mask):
        if value is None:
            return None
        if valid_mask is None:
            return value.reshape(-1)
        mask = valid_mask.to(torch.bool)
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        try:
            mask = torch.broadcast_to(mask, value.shape)
        except RuntimeError:
            mask = mask.expand_as(value)
        selected = value[mask]
        return selected.reshape(-1)

    def _student_entropy_target_enabled(self, target_name):
        if not self.distill_student_entropy_weight_enable:
            return False
        if self.distill_student_entropy_weight_target == 'gate_head_cls':
            return target_name in {'gate', 'head_cls'}
        return self.distill_student_entropy_weight_target == target_name

    @staticmethod
    def _normalized_entropy_dim(prob, dim=-1):
        classes = prob.shape[dim]
        if classes <= 1:
            out_shape = list(prob.shape)
            del out_shape[dim]
            return prob.new_zeros(out_shape)
        prob = prob.clamp_min(1e-6)
        entropy = -(prob * prob.log()).sum(dim=dim)
        return entropy / math.log(classes)

    def _compute_student_entropy_weight(
            self,
            student_prob,
            target_name,
            class_dim=-1,
            teacher_confidence=None,
            disagreement=None,
            valid_mask=None):
        if not self._student_entropy_target_enabled(target_name) or student_prob is None:
            return None, {}, self._zero(self.device)

        entropy = self._normalized_entropy_dim(student_prob, dim=class_dim)
        entropy_for_weight = entropy.detach() if self.distill_student_entropy_weight_detach else entropy

        teacher_confidence = teacher_confidence if teacher_confidence is not None else torch.ones_like(entropy_for_weight)
        disagreement = disagreement if disagreement is not None else torch.zeros_like(entropy_for_weight)
        max_prob = student_prob.max(dim=class_dim).values

        if self.distill_student_entropy_weight_mode == 'learnable' and self.student_entropy_weight_head is not None:
            features = torch.stack((
                entropy_for_weight,
                max_prob,
                teacher_confidence.detach(),
                disagreement.detach(),
            ), dim=-1)
            weight = self.student_entropy_weight_head(features)
        else:
            if self.distill_student_entropy_weight_formula == 'linear_map':
                raw_weight = (
                    self.distill_student_entropy_weight_min +
                    (self.distill_student_entropy_weight_max - self.distill_student_entropy_weight_min) * entropy_for_weight
                )
            elif self.distill_student_entropy_weight_formula == 'centered':
                raw_weight = 1.0 + self.distill_student_entropy_weight_beta * (
                    entropy_for_weight - self.distill_student_entropy_weight_tau)
            elif self.distill_student_entropy_weight_formula == 'asym_centered':
                pos = F.relu(entropy_for_weight - self.distill_student_entropy_weight_tau)
                neg = F.relu(self.distill_student_entropy_weight_tau - entropy_for_weight)
                raw_weight = 1.0 + self.distill_student_entropy_weight_beta_pos * pos - (
                    self.distill_student_entropy_weight_beta_neg * neg)
            else:
                raw_weight = 1.0 + self.distill_student_entropy_weight_beta * entropy_for_weight
            weight = raw_weight.clamp(
                min=self.distill_student_entropy_weight_min,
                max=self.distill_student_entropy_weight_max,
            )

        valid_weight = self._select_valid_values(weight.detach(), valid_mask)
        if self.distill_student_entropy_weight_normalize_mean and valid_weight is not None and valid_weight.numel():
            mean_weight = valid_weight.mean()
            weight = weight / (mean_weight.detach() + 1e-6)
            valid_weight = self._select_valid_values(weight.detach(), valid_mask)

        entropy_stats_value = self._select_valid_values(entropy.detach(), valid_mask)
        entropy_weight_std_value = valid_weight
        weight_mean = self._tensor_mean_or_zero(valid_weight, self.device)
        weight_std = self._tensor_std_or_zero(entropy_weight_std_value, self.device)

        reg_loss = self._zero(self.device)
        if self.distill_student_entropy_weight_reg > 0.0:
            reg_loss = (weight - 1.0).pow(2).mean() * self.distill_student_entropy_weight_reg

        stats = {
            'student_entropy_mean': self._tensor_mean_or_zero(entropy_stats_value, self.device),
            'student_entropy_weight_mean': weight_mean,
            'student_entropy_weight_std': weight_std,
            'student_entropy_weight_reg_loss': reg_loss.detach(),
            'student_entropy_weight_count': float(valid_weight.numel()) if valid_weight is not None else float(weight.numel()),
        }
        return weight, stats, reg_loss

    def _merge_student_entropy_stats(self, *stats_groups):
        merged = {}
        valid = [stats for stats in stats_groups if stats]
        if not valid:
            return merged

        total_count = sum(float(stats.get('student_entropy_weight_count', 0.0)) for stats in valid)
        if total_count <= 0.0:
            return merged

        keys = (
            'student_entropy_mean',
            'student_entropy_weight_mean',
            'student_entropy_weight_std',
            'student_entropy_weight_reg_loss',
            'final_kd_weight_mean',
        )
        for key in keys:
            value = None
            for stats in valid:
                count = float(stats.get('student_entropy_weight_count', 0.0))
                if count <= 0.0 or key not in stats:
                    continue
                weighted = stats[key] * count
                value = weighted if value is None else value + weighted
            if value is not None:
                merged[key] = value / total_count
        return merged

    def _collect_common_audit_stats(self):
        stats = {
            'fg_ratio': 0.0,
            'fg_count': 0.0,
            'gate_rgb_ratio': 0.0,
            'gate_ir_ratio': 0.0,
            'gate_entropy': 0.0,
            'align_conf_mean': 0.0,
        }
        student_cache = self._get_student_criterion_cache()
        fg_mask = student_cache.get('fg_mask')
        if fg_mask is not None:
            stats['fg_ratio'] = float(fg_mask.float().mean().detach().cpu())
            stats['fg_count'] = float(fg_mask.sum().detach().cpu())

        fusion_metrics = {'w_rgb': [], 'w_ir': [], 'gate_entropy': [], 'align_conf': []}
        for module in self.student_module_pairs_fusion:
            cache = getattr(module, 'kd_cache', {}) or {}
            if cache.get('w_rgb') is not None:
                fusion_metrics['w_rgb'].append(cache['w_rgb'].mean().detach())
            if cache.get('w_ir') is not None:
                fusion_metrics['w_ir'].append(cache['w_ir'].mean().detach())
            if cache.get('gate_entropy') is not None:
                fusion_metrics['gate_entropy'].append(cache['gate_entropy'].mean().detach())
            elif cache.get('spatial_logits') is not None:
                gate_probs = torch.softmax(cache['spatial_logits'], dim=1)
                gate_entropy = -(
                    gate_probs.clamp_min(1e-6) * gate_probs.clamp_min(1e-6).log()
                ).sum(dim=1, keepdim=True) / math.log(max(gate_probs.shape[1], 2))
                fusion_metrics['gate_entropy'].append(gate_entropy.mean().detach())
            if cache.get('align_conf') is not None:
                fusion_metrics['align_conf'].append(cache['align_conf'].mean().detach())

        if fusion_metrics['w_rgb']:
            stats['gate_rgb_ratio'] = float(torch.stack(fusion_metrics['w_rgb']).mean().cpu())
        if fusion_metrics['w_ir']:
            stats['gate_ir_ratio'] = float(torch.stack(fusion_metrics['w_ir']).mean().cpu())
        if fusion_metrics['gate_entropy']:
            stats['gate_entropy'] = float(torch.stack(fusion_metrics['gate_entropy']).mean().cpu())
        if fusion_metrics['align_conf']:
            stats['align_conf_mean'] = float(torch.stack(fusion_metrics['align_conf']).mean().cpu())
        return stats

    def _resolve_effective_distill_flags(self, current_epoch=None):
        schedule_enabled = bool(self.distill_schedule_enable)
        late_active = (
            self.distill_late_kd_start_epoch >= 0
            and current_epoch is not None
            and int(current_epoch) >= self.distill_late_kd_start_epoch
        )
        mid_active = (
            schedule_enabled
            and self.distill_mid_kd_start_epoch >= 0
            and current_epoch is not None
            and int(current_epoch) >= self.distill_mid_kd_start_epoch
            and not late_active
        )
        effective_normal = self.normal_distillation
        effective_cross = self.cross_attention
        effective_disable_gate = self.distill_disable_gate_kd
        effective_disable_pseudo = self.distill_disable_pseudo_fusion_kd
        effective_head_kd_policy = self.distill_head_kd_policy
        schedule_stage_id = float(self.SCHEDULE_STAGE_IDS['base'])

        if mid_active and self.distill_mid_kd_policy == 'normal_only_gate_off':
            effective_cross = False
            effective_disable_gate = True
            schedule_stage_id = float(self.SCHEDULE_STAGE_IDS['mid'])

        if late_active:
            if self.distill_late_kd_policy == 'gate_off':
                effective_disable_gate = True
            elif self.distill_late_kd_policy == 'normal_gate_off':
                effective_cross = False
                effective_disable_gate = True
            elif self.distill_late_kd_policy == 'strict_only_normal':
                effective_cross = False
                effective_disable_gate = True
                effective_disable_pseudo = True
            elif self.distill_late_kd_policy == 'strict_head_kd_off':
                effective_cross = False
                effective_disable_gate = True
                effective_disable_pseudo = True
                effective_head_kd_policy = 'off'
            if schedule_enabled:
                schedule_stage_id = float(self.SCHEDULE_STAGE_IDS['late'])

        return {
            'schedule_enabled': schedule_enabled,
            'mid_active': mid_active,
            'late_active': late_active,
            'schedule_stage_id': schedule_stage_id,
            'effective_normal': effective_normal,
            'effective_cross': effective_cross,
            'effective_disable_gate': effective_disable_gate,
            'effective_disable_pseudo': effective_disable_pseudo,
            'effective_head_kd_policy': effective_head_kd_policy,
            'mid_kd_policy_id': float(self.MID_KD_POLICY_IDS[self.distill_mid_kd_policy]),
            'late_kd_policy_id': float(self.LATE_KD_POLICY_IDS[self.distill_late_kd_policy]),
        }

    def _effective_flag_stats(self, current_epoch=None):
        effective = self._resolve_effective_distill_flags(current_epoch=current_epoch)
        effective_head_kd_policy = str(effective['effective_head_kd_policy'])
        head_cls_enabled = float(effective_head_kd_policy in ('full', 'cls_only'))
        head_dfl_enabled = float(effective_head_kd_policy in ('full', 'geom_only'))
        head_angle_enabled = float(effective_head_kd_policy in ('full', 'geom_only'))
        gate_enabled = float(
            self.enable_finegrained
            and not self.distill_only_normal
            and not effective['effective_disable_gate']
        )
        pseudo_enabled = float(
            self.enable_finegrained
            and not self.distill_only_normal
            and not effective['effective_disable_pseudo']
        )
        return {
            'gate_kd_enabled': gate_enabled,
            'pseudo_fusion_enabled': pseudo_enabled,
            'distill_gate_kd_enabled': gate_enabled,
            'distill_pseudo_fusion_enabled': pseudo_enabled,
            'head_cls_kd_enabled': head_cls_enabled,
            'head_dfl_kd_enabled': head_dfl_enabled,
            'head_angle_kd_enabled': head_angle_enabled,
            'head_kd_policy_id': float(self.HEAD_KD_POLICY_IDS[effective_head_kd_policy]),
            'effective_normal_enabled': float(effective['effective_normal']),
            'cross_attention_enabled': float(effective['effective_cross'] and not self.distill_only_normal),
            'mid_kd_active': float(effective['mid_active']),
            'late_kd_active': float(effective['late_active']),
            'schedule_stage_id': float(effective['schedule_stage_id']),
            'mid_kd_policy_id': effective['mid_kd_policy_id'],
            'late_kd_policy_id': effective['late_kd_policy_id'],
            'gate_kd_mode_is_normalized': float(self.distill_gate_kd_mode == 'normalized'),
            'gate_kd_weight': float(self.distill_gate_kd_weight),
            'gate_kd_conf_thr': float(self.distill_gate_kd_conf_thr),
        }

    def _reset_student_kd_cache(self):
        criterion = getattr(self.student_model, 'criterion', None)
        if criterion is not None and hasattr(criterion, 'kd_cache'):
            criterion.kd_cache = {}
        for module in self.student_module_pairs_fusion:
            if hasattr(module, 'kd_cache'):
                module.kd_cache = {}

    def parse_teacher_obb_predictions(self, preds, model):
        if preds is None:
            return None
        raw = preds[1] if isinstance(preds, tuple) and len(preds) == 2 else preds
        if not (isinstance(raw, tuple) and len(raw) == 2):
            raise TypeError('Expected teacher OBB forward output to contain raw feature maps and angle predictions.')
        feats, pred_angle = raw
        head = self._unwrap_model(model).model[-1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], head.no, -1) for xi in feats], 2).split(
            (head.reg_max * 4, head.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        confidence_maps = []
        score_maps = []
        for feat in feats:
            _, score_map = feat.split((head.reg_max * 4, head.nc), 1)
            score_maps.append(score_map)
            confidence_maps.append(score_map.sigmoid().amax(dim=1, keepdim=True))

        return {
            'feats': feats,
            'pred_scores': pred_scores,
            'pred_distri': pred_distri,
            'pred_angle': pred_angle,
            'confidence_maps': confidence_maps,
            'score_maps': score_maps,
            'reg_max': head.reg_max,
            'nc': head.nc,
        }

    def _teacher_pair_conf_probs(self, conf_rgb, conf_ir, target_hw):
        rgb = F.interpolate(conf_rgb, size=target_hw, mode='bilinear', align_corners=False)
        ir = F.interpolate(conf_ir, size=target_hw, mode='bilinear', align_corners=False)
        stacked = torch.cat((rgb, ir), dim=1).clamp_min(1e-6)
        return stacked / stacked.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _compute_gate_kd_loss(self, spatial_logits, conf_rgb, conf_ir):
        teacher_gate = self._teacher_pair_conf_probs(conf_rgb, conf_ir, spatial_logits.shape[2:]).detach().clamp_min(1e-6)
        if self.distill_gate_kd_mode == 'legacy':
            if self._student_entropy_target_enabled('gate'):
                log_p = F.log_softmax(spatial_logits, dim=1)
                kl_map = F.kl_div(log_p, teacher_gate, reduction='none').sum(dim=1, keepdim=True)
                teacher_conf = torch.maximum(
                    F.interpolate(conf_rgb, size=spatial_logits.shape[2:], mode='bilinear', align_corners=False),
                    F.interpolate(conf_ir, size=spatial_logits.shape[2:], mode='bilinear', align_corners=False),
                ).detach().clamp(0, 1)
                gate_probs = torch.softmax(spatial_logits, dim=1)
                disagreement = (gate_probs - teacher_gate).abs().mean(dim=1)
                entropy_weight, entropy_stats, reg_loss = self._compute_student_entropy_weight(
                    gate_probs,
                    target_name='gate',
                    class_dim=1,
                    teacher_confidence=teacher_conf.squeeze(1),
                    disagreement=disagreement,
                    valid_mask=(teacher_conf.squeeze(1) > 0),
                )
                combined_weight = teacher_conf * entropy_weight.unsqueeze(1)
                if float(combined_weight.sum().detach().cpu()) <= 0.0:
                    loss_raw = kl_map.new_zeros(())
                else:
                    loss_raw = (kl_map * combined_weight).sum() / combined_weight.sum().clamp_min(1e-6)
                entropy_stats['final_kd_weight_mean'] = self._tensor_mean_or_zero(
                    self._select_valid_values(combined_weight.detach(), teacher_conf > 0), self.device)
                loss = loss_raw * self.distill_gate_kd_weight + reg_loss
                return loss, {
                    'gate_kd_raw_normalized': loss_raw,
                    'gate_kd_valid_ratio': (teacher_conf > 0).to(kl_map.dtype).mean().detach(),
                    'gate_kd_weight': loss_raw.new_tensor(float(self.distill_gate_kd_weight)),
                    'gate_kd_conf_thr': loss_raw.new_tensor(float(self.distill_gate_kd_conf_thr)),
                    **entropy_stats,
                }
            loss_raw = F.kl_div(
                F.log_softmax(spatial_logits, dim=1),
                teacher_gate,
                reduction='batchmean'
            )
            loss = loss_raw * self.distill_gate_kd_weight
            return loss, {
                'gate_kd_raw_normalized': loss_raw,
                'gate_kd_valid_ratio': loss_raw.new_tensor(1.0),
                'gate_kd_weight': loss_raw.new_tensor(float(self.distill_gate_kd_weight)),
                'gate_kd_conf_thr': loss_raw.new_tensor(float(self.distill_gate_kd_conf_thr)),
            }

        teacher_gate = teacher_gate / teacher_gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
        temperature = max(float(self.distill_gate_kd_temperature), 1e-6)
        log_p = F.log_softmax(spatial_logits / temperature, dim=1)
        kl_map = F.kl_div(log_p, teacher_gate, reduction='none').sum(dim=1, keepdim=True) * (temperature * temperature)

        rgb_conf = F.interpolate(conf_rgb, size=spatial_logits.shape[2:], mode='bilinear', align_corners=False)
        ir_conf = F.interpolate(conf_ir, size=spatial_logits.shape[2:], mode='bilinear', align_corners=False)
        teacher_conf = torch.maximum(rgb_conf, ir_conf).detach().clamp(0, 1)

        entropy_stats = {}
        reg_loss = self._zero(self.device)
        entropy_weight = None
        if self._student_entropy_target_enabled('gate'):
            gate_probs = torch.softmax(spatial_logits, dim=1)
            disagreement = (gate_probs - teacher_gate).abs().mean(dim=1)
            if self.distill_gate_kd_mask_mode == 'conf_binary':
                valid_mask = teacher_conf.squeeze(1) >= self.distill_gate_kd_conf_thr
            elif self.distill_gate_kd_mask_mode == 'conf_binary_soft':
                valid_mask = teacher_conf.squeeze(1) >= self.distill_gate_kd_conf_thr
            else:
                valid_mask = teacher_conf.squeeze(1) > 0
            entropy_weight, entropy_stats, reg_loss = self._compute_student_entropy_weight(
                gate_probs,
                target_name='gate',
                class_dim=1,
                teacher_confidence=teacher_conf.squeeze(1),
                disagreement=disagreement,
                valid_mask=valid_mask,
            )

        if self.distill_gate_kd_mask_mode == 'none':
            if entropy_weight is None:
                loss_raw = kl_map.mean()
                valid_ratio = kl_map.new_tensor(1.0)
                final_weight = None
            else:
                weight = teacher_conf * entropy_weight.unsqueeze(1)
                valid_ratio = (teacher_conf > 0).to(kl_map.dtype).mean().detach()
                loss_raw = (kl_map * weight).sum() / weight.sum().clamp_min(1e-6)
                final_weight = weight
        elif self.distill_gate_kd_mask_mode == 'conf_binary':
            weight = (teacher_conf >= self.distill_gate_kd_conf_thr).to(kl_map.dtype)
            if entropy_weight is not None:
                weight = weight * entropy_weight.unsqueeze(1)
            valid_ratio = weight.mean().detach()
            loss_raw = (kl_map * weight).sum() / weight.sum().clamp_min(1.0)
            final_weight = weight
        elif self.distill_gate_kd_mask_mode == 'conf_soft':
            weight = teacher_conf
            if entropy_weight is not None:
                weight = weight * entropy_weight.unsqueeze(1)
            valid_ratio = (weight > 0).to(kl_map.dtype).mean().detach()
            loss_raw = (kl_map * weight).sum() / weight.sum().clamp_min(1e-6)
            final_weight = weight
        else:
            binary = (teacher_conf >= self.distill_gate_kd_conf_thr).to(kl_map.dtype)
            weight = binary * teacher_conf
            if entropy_weight is not None:
                weight = weight * entropy_weight.unsqueeze(1)
            valid_ratio = binary.mean().detach()
            if float(weight.sum().detach().cpu()) <= 0.0:
                loss_raw = kl_map.new_zeros(())
            else:
                loss_raw = (kl_map * weight).sum() / weight.sum().clamp_min(1e-6)
            final_weight = weight

        if entropy_weight is not None and final_weight is not None:
            entropy_stats['final_kd_weight_mean'] = self._tensor_mean_or_zero(
                self._select_valid_values(final_weight.detach(), final_weight > 0), self.device)
        loss = loss_raw * self.distill_gate_kd_weight + reg_loss
        return loss, {
            'gate_kd_raw_normalized': loss_raw,
            'gate_kd_valid_ratio': valid_ratio,
            'gate_kd_weight': loss_raw.new_tensor(float(self.distill_gate_kd_weight)),
            'gate_kd_conf_thr': loss_raw.new_tensor(float(self.distill_gate_kd_conf_thr)),
            **entropy_stats,
        }

    def _compute_fusion_kd(self, teacher_rgb_info, teacher_ir_info, effective_disable_gate=False,
                           effective_disable_pseudo=False):
        if not self.enable_finegrained or self.D_loss_fn_fusion is None:
            return self._zero(self.device), self._zero(self.device), {}

        pseudo_targets = []
        gate_loss = self._zero(self.device)
        gate_stats_accumulator = {
            'gate_kd_raw_normalized': [],
            'gate_kd_valid_ratio': [],
            'gate_kd_weight': [],
            'gate_kd_conf_thr': [],
            'student_entropy_mean': [],
            'student_entropy_weight_mean': [],
            'student_entropy_weight_std': [],
            'student_entropy_weight_reg_loss': [],
        }
        fusion_caches = [getattr(module, 'kd_cache', {}) for module in self.student_module_pairs_fusion]

        for idx, (student_feat, cache) in enumerate(zip(self.student_outputs_fusion, fusion_caches)):
            teacher_rgb_feat = self.teacher_outputs_rgb[idx + 1]
            teacher_ir_feat = self.teacher_outputs_ir[idx + 1]
            conf_rgb = teacher_rgb_info['confidence_maps'][idx]
            conf_ir = teacher_ir_info['confidence_maps'][idx]
            weights = self._teacher_pair_conf_probs(conf_rgb, conf_ir, teacher_rgb_feat.shape[2:])
            if not effective_disable_pseudo:
                pseudo_targets.append(weights[:, 0:1] * teacher_rgb_feat + weights[:, 1:2] * teacher_ir_feat)

            spatial_logits = cache.get('spatial_logits') if cache else None
            if spatial_logits is None or effective_disable_gate:
                continue
            layer_gate_loss, layer_gate_stats = self._compute_gate_kd_loss(spatial_logits, conf_rgb, conf_ir)
            gate_loss = gate_loss + layer_gate_loss
            for key, value in layer_gate_stats.items():
                gate_stats_accumulator.setdefault(key, []).append(value)

        fusion_loss = self.D_loss_fn_fusion(y_t=pseudo_targets, y_s=self.student_outputs_fusion) if pseudo_targets else self._zero(self.device)
        gate_stats = {}
        for key, values in gate_stats_accumulator.items():
            if values:
                gate_stats[key] = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                                               for v in values]).mean()
        return fusion_loss, gate_loss, gate_stats

    def _compute_head_kd(self, teacher_rgb_info, teacher_ir_info, effective_head_kd_policy='full'):
        student_cache = self._get_student_criterion_cache()
        required_keys = ('pred_scores', 'pred_distri', 'pred_angle', 'fg_mask', 'target_scores')
        if any(k not in student_cache for k in required_keys):
            return self._zero(self.device), self._zero(self.device), self._zero(self.device), {}

        fg_mask = student_cache['fg_mask']
        if fg_mask is None or not fg_mask.any():
            return self._zero(self.device), self._zero(self.device), self._zero(self.device), {}

        student_scores = student_cache['pred_scores']
        student_distri = student_cache['pred_distri']
        student_angle = student_cache['pred_angle']
        sample_weight = student_cache['target_scores'].max(dim=-1, keepdim=True).values

        teacher_rgb_scores = teacher_rgb_info['pred_scores']
        teacher_ir_scores = teacher_ir_info['pred_scores']
        teacher_rgb_distri = teacher_rgb_info['pred_distri']
        teacher_ir_distri = teacher_ir_info['pred_distri']
        teacher_rgb_angle = teacher_rgb_info['pred_angle']
        teacher_ir_angle = teacher_ir_info['pred_angle']

        conf_rgb = teacher_rgb_scores.sigmoid().amax(dim=-1, keepdim=True)
        conf_ir = teacher_ir_scores.sigmoid().amax(dim=-1, keepdim=True)
        weight_sum = (conf_rgb + conf_ir).clamp_min(1e-6)
        w_rgb = conf_rgb / weight_sum
        w_ir = conf_ir / weight_sum

        cls_enabled = effective_head_kd_policy in ('full', 'cls_only')
        dfl_enabled = effective_head_kd_policy in ('full', 'geom_only')
        angle_enabled = effective_head_kd_policy in ('full', 'geom_only')

        cls_loss = self._zero(self.device)
        dfl_loss = self._zero(self.device)
        angle_loss = self._zero(self.device)
        head_stats = {}

        if cls_enabled:
            cls_target = w_rgb * teacher_rgb_scores.sigmoid() + w_ir * teacher_ir_scores.sigmoid()
            cls_loss = F.binary_cross_entropy_with_logits(student_scores, cls_target, reduction='none')
            if self._student_entropy_target_enabled('head_cls'):
                student_cls_prob = torch.softmax(student_scores, dim=-1)
                teacher_conf = torch.maximum(conf_rgb, conf_ir).detach().squeeze(-1)
                disagreement = (student_scores.sigmoid() - cls_target.detach()).abs().mean(dim=-1)
                entropy_weight, head_stats, reg_loss = self._compute_student_entropy_weight(
                    student_cls_prob,
                    target_name='head_cls',
                    class_dim=-1,
                    teacher_confidence=teacher_conf,
                    disagreement=disagreement,
                    valid_mask=fg_mask,
                )
                cls_weight = sample_weight * teacher_conf.unsqueeze(-1) * entropy_weight.unsqueeze(-1)
                head_stats['final_kd_weight_mean'] = self._tensor_mean_or_zero(
                    self._select_valid_values(cls_weight.detach(), fg_mask), self.device)
                cls_loss = self._weighted_masked_mean(cls_loss, fg_mask, cls_weight) + reg_loss
            else:
                cls_loss = self._weighted_masked_mean(cls_loss, fg_mask, sample_weight)

        if dfl_enabled:
            reg_max = teacher_rgb_info['reg_max']
            student_bins = student_distri.view(student_distri.shape[0], student_distri.shape[1], 4, reg_max)
            teacher_rgb_bins = teacher_rgb_distri.view(teacher_rgb_distri.shape[0], teacher_rgb_distri.shape[1], 4, reg_max)
            teacher_ir_bins = teacher_ir_distri.view(teacher_ir_distri.shape[0], teacher_ir_distri.shape[1], 4, reg_max)
            teacher_bins = (
                w_rgb.unsqueeze(-1) * teacher_rgb_bins.softmax(dim=-1)
                + w_ir.unsqueeze(-1) * teacher_ir_bins.softmax(dim=-1)
            ).clamp_min(1e-6)
            dfl_loss = F.kl_div(F.log_softmax(student_bins, dim=-1), teacher_bins, reduction='none').sum(dim=-1)
            dfl_loss = self._weighted_masked_mean(dfl_loss, fg_mask, sample_weight)

        if angle_enabled:
            sin_term = w_rgb * torch.sin(2.0 * teacher_rgb_angle) + w_ir * torch.sin(2.0 * teacher_ir_angle)
            cos_term = w_rgb * torch.cos(2.0 * teacher_rgb_angle) + w_ir * torch.cos(2.0 * teacher_ir_angle)
            teacher_angle = 0.5 * torch.atan2(sin_term, cos_term)
            angle_loss = 1.0 - torch.cos(2.0 * (student_angle - teacher_angle))
            angle_loss = self._weighted_masked_mean(angle_loss, fg_mask, sample_weight)
        return cls_loss, dfl_loss, angle_loss, head_stats

    @staticmethod
    def _normalized_entropy(prob):
        if prob.shape[-1] <= 1:
            return torch.zeros(prob.shape[:-1], device=prob.device, dtype=prob.dtype)
        prob = prob.clamp_min(1e-6)
        entropy = -(prob * prob.log()).sum(dim=-1)
        return entropy / math.log(prob.shape[-1])

    @staticmethod
    def _jsd(prob_a, prob_b):
        prob_a = prob_a.clamp_min(1e-6)
        prob_b = prob_b.clamp_min(1e-6)
        mean_prob = 0.5 * (prob_a + prob_b)
        return 0.5 * (
            (prob_a * (prob_a.log() - mean_prob.log())).sum(dim=-1) +
            (prob_b * (prob_b.log() - mean_prob.log())).sum(dim=-1)
        )

    @staticmethod
    def build_stage2_teacher_cls_targets(
            student_fg_mask,
            teacher_logits_rgb,
            teacher_logits_ir,
            temperature=2.0,
            conf_thr=0.45,
            entropy_thr=0.35,
            jsd_thr=0.10):
        base_prob_rgb = F.softmax(teacher_logits_rgb, dim=-1)
        base_prob_ir = F.softmax(teacher_logits_ir, dim=-1)
        temp_prob_rgb = F.softmax(teacher_logits_rgb / temperature, dim=-1)
        temp_prob_ir = F.softmax(teacher_logits_ir / temperature, dim=-1)

        rgb_max_prob = base_prob_rgb.max(dim=-1).values
        ir_max_prob = base_prob_ir.max(dim=-1).values
        rgb_entropy = Multimodal_Distillation_loss._normalized_entropy(base_prob_rgb)
        ir_entropy = Multimodal_Distillation_loss._normalized_entropy(base_prob_ir)
        rgb_reliability = rgb_max_prob * (1.0 - rgb_entropy)
        ir_reliability = ir_max_prob * (1.0 - ir_entropy)

        rgb_reliable_mask = (rgb_max_prob >= conf_thr) & (rgb_entropy <= entropy_thr)
        ir_reliable_mask = (ir_max_prob >= conf_thr) & (ir_entropy <= entropy_thr)
        jsd = Multimodal_Distillation_loss._jsd(base_prob_rgb, base_prob_ir)

        both_reliable_mask = rgb_reliable_mask & ir_reliable_mask
        consensus_mask = both_reliable_mask & (jsd < jsd_thr)
        disagreement_mask = both_reliable_mask & ~consensus_mask
        rgb_better_mask = rgb_reliability >= ir_reliability
        rgb_select_mask = (rgb_reliable_mask & ~ir_reliable_mask) | (disagreement_mask & rgb_better_mask)
        ir_select_mask = (ir_reliable_mask & ~rgb_reliable_mask) | (disagreement_mask & ~rgb_better_mask)
        selected_mask = student_fg_mask & (consensus_mask | rgb_select_mask | ir_select_mask)

        target_probs = torch.zeros_like(temp_prob_rgb)
        consensus_selected = selected_mask & consensus_mask
        rgb_selected = selected_mask & rgb_select_mask
        ir_selected = selected_mask & ir_select_mask

        if consensus_selected.any():
            reliability_sum = (rgb_reliability + ir_reliability).clamp_min(1e-6)
            rgb_weight = (rgb_reliability / reliability_sum).unsqueeze(-1)
            ir_weight = (ir_reliability / reliability_sum).unsqueeze(-1)
            target_probs[consensus_selected] = (
                rgb_weight * temp_prob_rgb + ir_weight * temp_prob_ir
            )[consensus_selected]
        if rgb_selected.any():
            target_probs[rgb_selected] = temp_prob_rgb[rgb_selected]
        if ir_selected.any():
            target_probs[ir_selected] = temp_prob_ir[ir_selected]

        final_reliability = torch.zeros_like(rgb_reliability)
        if consensus_selected.any():
            reliability_sum = (rgb_reliability + ir_reliability).clamp_min(1e-6)
            consensus_reliability = (
                (rgb_reliability.square() + ir_reliability.square()) / reliability_sum
            )
            final_reliability[consensus_selected] = consensus_reliability[consensus_selected]
        if rgb_selected.any():
            final_reliability[rgb_selected] = rgb_reliability[rgb_selected]
        if ir_selected.any():
            final_reliability[ir_selected] = ir_reliability[ir_selected]

        return {
            'selected_mask': selected_mask,
            'target_probs': target_probs,
            'rgb_reliable_mask': rgb_reliable_mask,
            'ir_reliable_mask': ir_reliable_mask,
            'rgb_reliability': rgb_reliability,
            'ir_reliability': ir_reliability,
            'teacher_reliability': final_reliability,
            'jsd': jsd,
        }

    @staticmethod
    def compute_stage2_cls_kd_loss(student_logits, teacher_target_probs, selected_mask, temperature=2.0):
        if selected_mask is None or not selected_mask.any():
            return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        loss = F.kl_div(log_probs, teacher_target_probs, reduction='none').sum(dim=-1)
        selected_weight = selected_mask.to(loss.dtype)
        return (loss * selected_weight).sum() / selected_weight.sum().clamp_min(1.0) * (temperature ** 2)

    def _compute_stage2_cls_only_kd(self, teacher_rgb_info, teacher_ir_info):
        student_cache = self._get_student_criterion_cache()
        required_keys = ('pred_scores', 'fg_mask')
        if any(k not in student_cache for k in required_keys):
            return self._zero(self.device)

        stage2_targets = self.build_stage2_teacher_cls_targets(
            student_fg_mask=student_cache['fg_mask'],
            teacher_logits_rgb=teacher_rgb_info['pred_scores'],
            teacher_logits_ir=teacher_ir_info['pred_scores'],
            temperature=self.distill_kd_temperature,
            conf_thr=self.teacher_conf_thr,
            entropy_thr=self.teacher_entropy_thr,
            jsd_thr=self.teacher_jsd_thr,
        )
        cls_kd_loss = self.compute_stage2_cls_kd_loss(
            student_logits=student_cache['pred_scores'],
            teacher_target_probs=stage2_targets['target_probs'],
            selected_mask=stage2_targets['selected_mask'],
            temperature=self.distill_kd_temperature,
        )

        selected_mask = stage2_targets['selected_mask']
        selected_count = selected_mask.sum().to(dtype=torch.float32)
        denom = selected_count.clamp_min(1.0)
        rgb_selected_ratio = (selected_mask & stage2_targets['rgb_reliable_mask']).sum().to(torch.float32) / denom
        ir_selected_ratio = (selected_mask & stage2_targets['ir_reliable_mask']).sum().to(torch.float32) / denom
        if selected_mask.any():
            avg_teacher_reliability = stage2_targets['teacher_reliability'][selected_mask].mean()
            std_teacher_reliability = self._tensor_std_or_zero(
                stage2_targets['teacher_reliability'][selected_mask],
                self.device,
            )
            avg_jsd_on_selected = stage2_targets['jsd'][selected_mask].mean()
            std_jsd_on_selected = self._tensor_std_or_zero(stage2_targets['jsd'][selected_mask], self.device)
        else:
            avg_teacher_reliability = self._zero(self.device)
            std_teacher_reliability = self._zero(self.device)
            avg_jsd_on_selected = self._zero(self.device)
            std_jsd_on_selected = self._zero(self.device)

        self._update_last_loss_dict({
            'cls_kd_loss': cls_kd_loss,
            'num_selected_pos': selected_count,
            'rgb_selected_ratio': rgb_selected_ratio,
            'ir_selected_ratio': ir_selected_ratio,
            'avg_teacher_reliability': avg_teacher_reliability,
            'std_teacher_reliability': std_teacher_reliability,
            'avg_jsd_on_selected': avg_jsd_on_selected,
            'std_jsd_on_selected': std_jsd_on_selected,
        })
        self.last_loss_dict.update(self._collect_common_audit_stats())
        return cls_kd_loss

    def get_loss(self, teacher_preds_rgb=None, teacher_preds_ir=None, batch=None, current_epoch=None):
        effective_stats = self._effective_flag_stats(current_epoch=current_epoch)
        if self.stage2_cls_only:
            self.device = self._get_model_device(self.student_model)
            cls_kd_loss = self._zero(self.device)
            if teacher_preds_rgb is not None and teacher_preds_ir is not None:
                teacher_rgb_info = self.parse_teacher_obb_predictions(teacher_preds_rgb, self.teacher_model_rgb)
                teacher_ir_info = self.parse_teacher_obb_predictions(teacher_preds_ir, self.teacher_model_ir)
                cls_kd_loss = self._compute_stage2_cls_only_kd(teacher_rgb_info, teacher_ir_info)
            quant_loss = torch.stack((cls_kd_loss, self._zero(self.device)))
            self.last_loss_dict.update({
                'branch_rgb_kd': 0.0,
                'branch_ir_kd': 0.0,
                'fusion_kd': 0.0,
                'gate_kd': 0.0,
                'cross_ir_to_rgb_kd': 0.0,
                'cross_rgb_to_ir_kd': 0.0,
                'head_cls_kd': float(cls_kd_loss.detach().cpu()),
                'head_dfl_kd': 0.0,
                'head_angle_kd': 0.0,
                'd_loss_total': float(cls_kd_loss.detach().cpu()),
                'c_loss_total': 0.0,
                'normal_distill_loss': float(cls_kd_loss.detach().cpu()),
                'cross_distill_loss': 0.0,
                'gate_kd_raw_normalized': 0.0,
                'gate_kd_valid_ratio': 0.0,
                'student_entropy_mean': 0.0,
                'student_entropy_weight_mean': 0.0,
                'student_entropy_weight_std': 0.0,
                'final_kd_weight_mean': 0.0,
                'student_entropy_weight_reg_loss': 0.0,
                **effective_stats,
            })
            self._reset_student_kd_cache()
            return quant_loss

        self._validate_captured_features()
        self.device = self._get_model_device(self.student_model)
        quant_loss = torch.zeros(2, device=self.device)
        branch_rgb_loss = self._zero(self.device)
        branch_ir_loss = self._zero(self.device)
        cross_ir_to_rgb_loss = self._zero(self.device)
        cross_rgb_to_ir_loss = self._zero(self.device)
        fusion_loss = self._zero(self.device)
        gate_loss = self._zero(self.device)
        head_cls_loss = self._zero(self.device)
        head_dfl_loss = self._zero(self.device)
        head_angle_loss = self._zero(self.device)
        gate_stats = {}
        head_stats = {}
        effective = self._resolve_effective_distill_flags(current_epoch=current_epoch)

        if effective['effective_normal']:
            branch_rgb_loss = self.D_loss_fn_rgb(y_t=self.teacher_outputs_rgb, y_s=self.student_outputs_rgb)
            branch_ir_loss = self.D_loss_fn_ir(y_t=self.teacher_outputs_ir, y_s=self.student_outputs_ir)
            quant_loss[0] += branch_rgb_loss + branch_ir_loss
        if effective['effective_cross'] and not self.distill_only_normal:
            cross_ir_to_rgb_loss = self.Cross_loss_ir_to_rgb(y_t=self.teacher_outputs_ir, y_s=self.student_outputs_rgb)
            cross_rgb_to_ir_loss = self.Cross_loss_rgb_to_ir(y_t=self.teacher_outputs_rgb, y_s=self.student_outputs_ir)
            quant_loss[1] += cross_ir_to_rgb_loss + cross_rgb_to_ir_loss

        teacher_rgb_info = teacher_ir_info = None
        if (
                self.enable_finegrained
                and not self.distill_only_normal
                and teacher_preds_rgb is not None
                and teacher_preds_ir is not None):
            teacher_rgb_info = self.parse_teacher_obb_predictions(teacher_preds_rgb, self.teacher_model_rgb)
            teacher_ir_info = self.parse_teacher_obb_predictions(teacher_preds_ir, self.teacher_model_ir)
            fusion_loss, gate_loss, gate_stats = self._compute_fusion_kd(
                teacher_rgb_info,
                teacher_ir_info,
                effective_disable_gate=effective['effective_disable_gate'],
                effective_disable_pseudo=effective['effective_disable_pseudo'],
            )
            head_cls_loss, head_dfl_loss, head_angle_loss, head_stats = self._compute_head_kd(
                teacher_rgb_info,
                teacher_ir_info,
                effective_head_kd_policy=effective['effective_head_kd_policy'],
            )
            quant_loss[0] += fusion_loss + gate_loss
            quant_loss[1] += head_cls_loss + head_dfl_loss + head_angle_loss

        entropy_stats = self._merge_student_entropy_stats(gate_stats, head_stats)

        self._update_last_loss_dict({
            'branch_rgb_kd': branch_rgb_loss,
            'branch_ir_kd': branch_ir_loss,
            'fusion_kd': fusion_loss,
            'gate_kd': gate_loss,
            'cross_ir_to_rgb_kd': cross_ir_to_rgb_loss,
            'cross_rgb_to_ir_kd': cross_rgb_to_ir_loss,
            'head_cls_kd': head_cls_loss,
            'head_dfl_kd': head_dfl_loss,
            'head_angle_kd': head_angle_loss,
            'd_loss_total': quant_loss[0],
            'c_loss_total': quant_loss[1],
            'normal_distill_loss': quant_loss[0],
            'cross_distill_loss': quant_loss[1],
            'distill_only_normal_enabled': float(self.distill_only_normal),
            'gate_kd_raw_normalized': gate_stats.get('gate_kd_raw_normalized', 0.0),
            'gate_kd_valid_ratio': gate_stats.get('gate_kd_valid_ratio', 0.0),
            'student_entropy_mean': entropy_stats.get('student_entropy_mean', 0.0),
            'student_entropy_weight_mean': entropy_stats.get('student_entropy_weight_mean', 0.0),
            'student_entropy_weight_std': entropy_stats.get('student_entropy_weight_std', 0.0),
            'final_kd_weight_mean': entropy_stats.get('final_kd_weight_mean', 0.0),
            'student_entropy_weight_reg_loss': entropy_stats.get('student_entropy_weight_reg_loss', 0.0),
            **effective_stats,
        })
        self.last_loss_dict.update(self._collect_common_audit_stats())

        self.teacher_outputs_rgb.clear()
        self.teacher_outputs_ir.clear()
        self.student_outputs_rgb.clear()
        self.student_outputs_ir.clear()
        self.student_outputs_fusion.clear()
        self._reset_student_kd_cache()
        return quant_loss

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
        self.remove_handle = []


class BaseTrainer:
    """
    BaseTrainer.

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        check_resume (method): Method to check if training should be resumed from a saved checkpoint.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        overrides = overrides or {}

        self.ddp_extra_overrides = {}
        self.teacher_model_rgb_path = overrides.pop('Teacher_Model_RGB_Path', None)
        self.teacher_model_ir_path = overrides.pop('Teacher_Model_IR_Path', None)
        self.teacher_model_path = overrides.pop('Teacher_Model_Path', None)
        self.distill_disable_all = bool(overrides.pop('distill_disable_all', False))
        self.distill_only_normal = bool(overrides.pop('distill_only_normal', False))
        self.distill_disable_gate_kd = bool(overrides.pop('distill_disable_gate_kd', False))
        self.distill_disable_pseudo_fusion_kd = bool(overrides.pop('distill_disable_pseudo_fusion_kd', False))
        self.distill_head_kd_policy = str(overrides.pop('distill_head_kd_policy', 'full'))
        self.distill_epoch_scale_mode = str(overrides.pop('distill_epoch_scale_mode', 'legacy_cosine'))
        self.distill_epoch_scale_start = float(overrides.pop('distill_epoch_scale_start', 1.0))
        self.distill_epoch_scale_end = float(overrides.pop('distill_epoch_scale_end', 0.1))
        self.distill_epoch_scale_decay_start = int(overrides.pop('distill_epoch_scale_decay_start', 0))
        self.distill_epoch_scale_decay_end = int(overrides.pop('distill_epoch_scale_decay_end', -1))
        self.distill_zero_after_epoch = int(overrides.pop('distill_zero_after_epoch', -1))
        self.distill_schedule_enable = bool(overrides.pop('distill_schedule_enable', False))
        self.distill_mid_kd_start_epoch = int(overrides.pop('distill_mid_kd_start_epoch', -1))
        self.distill_mid_kd_policy = str(overrides.pop('distill_mid_kd_policy', 'none'))
        self.distill_gate_kd_mode = str(overrides.pop('distill_gate_kd_mode', 'legacy'))
        self.distill_gate_kd_weight = float(overrides.pop('distill_gate_kd_weight', 1.0))
        self.distill_gate_kd_temperature = float(overrides.pop('distill_gate_kd_temperature', 1.0))
        self.distill_gate_kd_mask_mode = str(overrides.pop('distill_gate_kd_mask_mode', 'none'))
        self.distill_gate_kd_conf_thr = float(overrides.pop('distill_gate_kd_conf_thr', 0.25))
        self.distill_student_entropy_weight_enable = bool(
            overrides.pop('distill_student_entropy_weight_enable', False))
        self.distill_student_entropy_weight_mode = str(
            overrides.pop('distill_student_entropy_weight_mode', 'fixed'))
        self.distill_student_entropy_weight_formula = str(
            overrides.pop('distill_student_entropy_weight_formula', 'fixed_boost'))
        self.distill_student_entropy_weight_min = float(
            overrides.pop('distill_student_entropy_weight_min', 0.5))
        self.distill_student_entropy_weight_max = float(
            overrides.pop('distill_student_entropy_weight_max', 1.5))
        self.distill_student_entropy_weight_beta = float(
            overrides.pop('distill_student_entropy_weight_beta', 0.5))
        self.distill_student_entropy_weight_beta_pos = float(
            overrides.pop('distill_student_entropy_weight_beta_pos', 1.0))
        self.distill_student_entropy_weight_beta_neg = float(
            overrides.pop('distill_student_entropy_weight_beta_neg', 0.5))
        self.distill_student_entropy_weight_tau = float(
            overrides.pop('distill_student_entropy_weight_tau', 0.5))
        self.distill_student_entropy_weight_normalize_mean = bool(
            overrides.pop('distill_student_entropy_weight_normalize_mean', False))
        self.distill_student_entropy_weight_detach = bool(
            overrides.pop('distill_student_entropy_weight_detach', True))
        self.distill_student_entropy_weight_target = str(
            overrides.pop('distill_student_entropy_weight_target', 'gate'))
        self.distill_student_entropy_weight_reg = float(
            overrides.pop('distill_student_entropy_weight_reg', 0.0))
        self.distill_late_kd_start_epoch = int(overrides.pop('distill_late_kd_start_epoch', -1))
        self.distill_late_kd_policy = str(overrides.pop('distill_late_kd_policy', 'none'))
        self.distill_stage2_cls_only = bool(overrides.pop('distill_stage2_cls_only', False))
        self.distill_resume_ckpt = overrides.pop('distill_resume_ckpt', None)
        self.freeze_backbone_fusion = bool(overrides.pop('freeze_backbone_fusion', True))
        self.train_neck_head_only = bool(overrides.pop('train_neck_head_only', True))
        self.stage2_lr_mult = float(overrides.pop('stage2_lr_mult', 0.1))
        self.stage2_epochs = int(overrides.pop('stage2_epochs', 25))
        self.stage2_disable_early_stop = bool(overrides.pop('stage2_disable_early_stop', True))
        self.distill_cls_kd_weight = float(overrides.pop('distill_cls_kd_weight', 0.05))
        self.distill_kd_temperature = float(overrides.pop('distill_kd_temperature', 2.0))
        self.teacher_conf_thr = float(overrides.pop('teacher_conf_thr', 0.45))
        self.teacher_entropy_thr = float(overrides.pop('teacher_entropy_thr', 0.35))
        self.teacher_jsd_thr = float(overrides.pop('teacher_jsd_thr', 0.10))
        self.audit_mode = bool(overrides.pop('audit_mode', False))
        self.audit_disable_early_stop = bool(overrides.pop('audit_disable_early_stop', True))
        self.audit_ckpt_interval = int(overrides.pop('audit_ckpt_interval', 5))
        self.audit_probe_epochs = int(overrides.pop('audit_probe_epochs', 10))
        self.audit_probe_fractions = overrides.pop('audit_probe_fractions', '0.2,0.4,0.6,0.8')
        self.audit_probes = overrides.pop('audit_probes', 'continue,kd_off,normal_only,cross_off,lr_x0.3')
        self.audit_log_every_n = int(overrides.pop('audit_log_every_n', 200))
        self.audit_output_name = str(overrides.pop('audit_output_name', 'headroom_audit'))
        self.audit_eval_split = str(overrides.pop('audit_eval_split', 'val'))
        self.stage2_frozen_module_indices = []
        self.stage2_trainable_param_count = 0
        self.stage2_frozen_param_count = 0
        self.stage2_resume_ckpt_used = None
        self.audit_probe_registry = AuditProbeRegistry()
        self.audit_logger = None
        self.audit_epoch_stats = {}
        self.audit_epoch_grad_samples = []
        self.audit_grad_params = []
        self.audit_grad_param_name = ''
        self.audit_checkpoint_epochs = []
        self.audit_probe_fraction_values = []
        self.audit_saved_checkpoints = []

        stage2_overrides = {
            'distill_stage2_cls_only': self.distill_stage2_cls_only,
            'distill_resume_ckpt': self.distill_resume_ckpt,
            'freeze_backbone_fusion': self.freeze_backbone_fusion,
            'train_neck_head_only': self.train_neck_head_only,
            'stage2_lr_mult': self.stage2_lr_mult,
            'stage2_epochs': self.stage2_epochs,
            'stage2_disable_early_stop': self.stage2_disable_early_stop,
            'distill_cls_kd_weight': self.distill_cls_kd_weight,
            'distill_kd_temperature': self.distill_kd_temperature,
            'teacher_conf_thr': self.teacher_conf_thr,
            'teacher_entropy_thr': self.teacher_entropy_thr,
            'teacher_jsd_thr': self.teacher_jsd_thr,
        }
        distill_toggle_overrides = {
            'distill_only_normal': self.distill_only_normal,
            'distill_disable_gate_kd': self.distill_disable_gate_kd,
            'distill_disable_pseudo_fusion_kd': self.distill_disable_pseudo_fusion_kd,
            'distill_head_kd_policy': self.distill_head_kd_policy,
            'distill_epoch_scale_mode': self.distill_epoch_scale_mode,
            'distill_epoch_scale_start': self.distill_epoch_scale_start,
            'distill_epoch_scale_end': self.distill_epoch_scale_end,
            'distill_epoch_scale_decay_start': self.distill_epoch_scale_decay_start,
            'distill_epoch_scale_decay_end': self.distill_epoch_scale_decay_end,
            'distill_zero_after_epoch': self.distill_zero_after_epoch,
            'distill_schedule_enable': self.distill_schedule_enable,
            'distill_mid_kd_start_epoch': self.distill_mid_kd_start_epoch,
            'distill_mid_kd_policy': self.distill_mid_kd_policy,
            'distill_gate_kd_mode': self.distill_gate_kd_mode,
            'distill_gate_kd_weight': self.distill_gate_kd_weight,
            'distill_gate_kd_temperature': self.distill_gate_kd_temperature,
            'distill_gate_kd_mask_mode': self.distill_gate_kd_mask_mode,
            'distill_gate_kd_conf_thr': self.distill_gate_kd_conf_thr,
            'distill_student_entropy_weight_enable': self.distill_student_entropy_weight_enable,
            'distill_student_entropy_weight_mode': self.distill_student_entropy_weight_mode,
            'distill_student_entropy_weight_formula': self.distill_student_entropy_weight_formula,
            'distill_student_entropy_weight_min': self.distill_student_entropy_weight_min,
            'distill_student_entropy_weight_max': self.distill_student_entropy_weight_max,
            'distill_student_entropy_weight_beta': self.distill_student_entropy_weight_beta,
            'distill_student_entropy_weight_beta_pos': self.distill_student_entropy_weight_beta_pos,
            'distill_student_entropy_weight_beta_neg': self.distill_student_entropy_weight_beta_neg,
            'distill_student_entropy_weight_tau': self.distill_student_entropy_weight_tau,
            'distill_student_entropy_weight_normalize_mean': self.distill_student_entropy_weight_normalize_mean,
            'distill_student_entropy_weight_detach': self.distill_student_entropy_weight_detach,
            'distill_student_entropy_weight_target': self.distill_student_entropy_weight_target,
            'distill_student_entropy_weight_reg': self.distill_student_entropy_weight_reg,
            'distill_late_kd_start_epoch': self.distill_late_kd_start_epoch,
            'distill_late_kd_policy': self.distill_late_kd_policy,
        }
        audit_overrides = {
            'audit_mode': self.audit_mode,
            'audit_disable_early_stop': self.audit_disable_early_stop,
            'audit_ckpt_interval': self.audit_ckpt_interval,
            'audit_probe_epochs': self.audit_probe_epochs,
            'audit_probe_fractions': self.audit_probe_fractions,
            'audit_probes': self.audit_probes,
            'audit_log_every_n': self.audit_log_every_n,
            'audit_output_name': self.audit_output_name,
            'audit_eval_split': self.audit_eval_split,
        }

        self.distill_student_rgb_layers = overrides.pop('distill_student_rgb_layers', None)
        self.distill_student_ir_layers = overrides.pop('distill_student_ir_layers', None)
        self.distill_teacher_rgb_layers = overrides.pop('distill_teacher_rgb_layers', None)
        self.distill_teacher_ir_layers = overrides.pop('distill_teacher_ir_layers', None)

        self.distill_student_rgb_channels = overrides.pop('distill_student_rgb_channels', None)
        self.distill_student_ir_channels = overrides.pop('distill_student_ir_channels', None)
        self.distill_student_fusion_layers = overrides.pop('distill_student_fusion_layers', None)
        self.distill_student_fusion_channels = overrides.pop('distill_student_fusion_channels', None)
        self.distill_teacher_rgb_channels = overrides.pop('distill_teacher_rgb_channels', None)
        self.distill_teacher_ir_channels = overrides.pop('distill_teacher_ir_channels', None)

        self.distill_cross_attention = overrides.pop('distill_cross_attention', True)
        self.distill_normal_distillation = overrides.pop('distill_normal_distillation', True)

        if "Distillation" not in overrides:
            overrides["Distillation"] = None

        if overrides["Distillation"] is not None:  # 需要蒸馏
            self.Distillation = overrides["Distillation"]
            self.loss_type = overrides['loss_type']
            self.distill_weight = overrides['distill_weight']
            self.online = overrides['online']
            if self.Distillation == "MultiDistillation":
                # 多模态蒸馏, 有一个RGB教师和一个IR教师
                self.Teacher_Model_RGB = overrides.pop("Teacher_Model_RGB", None)
                self.Teacher_Model_IR = overrides.pop("Teacher_Model_IR", None)
                if self.Teacher_Model_RGB is not None and self.teacher_model_rgb_path is None:
                    self.teacher_model_rgb_path = getattr(self.Teacher_Model_RGB, 'pt_path', None)
                if self.Teacher_Model_IR is not None and self.teacher_model_ir_path is None:
                    self.teacher_model_ir_path = getattr(self.Teacher_Model_IR, 'pt_path', None)
                if self.Teacher_Model_RGB is None and self.teacher_model_rgb_path is not None:
                    self.Teacher_Model_RGB, _ = attempt_load_one_weight(str(self.teacher_model_rgb_path), device='cpu')
                if self.Teacher_Model_IR is None and self.teacher_model_ir_path is not None:
                    self.Teacher_Model_IR, _ = attempt_load_one_weight(str(self.teacher_model_ir_path), device='cpu')
                if self.Teacher_Model_RGB is None or self.Teacher_Model_IR is None:
                    raise ValueError(
                        'MultiDistillation requires Teacher_Model_RGB/IR objects or Teacher_Model_RGB_Path/IR_Path.'
                    )
            else:  # 单模态蒸馏
                self.Teacher_Model = overrides.pop("Teacher_Model", None)
                if self.Teacher_Model is not None and self.teacher_model_path is None:
                    self.teacher_model_path = getattr(self.Teacher_Model, 'pt_path', None)
                if self.Teacher_Model is None and self.teacher_model_path is not None:
                    self.Teacher_Model, _ = attempt_load_one_weight(str(self.teacher_model_path), device='cpu')
                if self.Teacher_Model is None:
                    raise ValueError('Distillation requires Teacher_Model object or Teacher_Model_Path.')

            overrides.pop("loss_type")
            overrides.pop("Distillation")
            overrides.pop("online")
            overrides.pop("distill_weight")

            ddp_overrides = {
                'Distillation': self.Distillation,
                'loss_type': self.loss_type,
                'distill_weight': self.distill_weight,
                'online': self.online,
                'distill_cross_attention': self.distill_cross_attention,
                'distill_normal_distillation': self.distill_normal_distillation,
                **distill_toggle_overrides,
                **stage2_overrides,
            }
            optional_ddp_keys = {
                'distill_student_rgb_layers': self.distill_student_rgb_layers,
                'distill_student_ir_layers': self.distill_student_ir_layers,
                'distill_teacher_rgb_layers': self.distill_teacher_rgb_layers,
                'distill_teacher_ir_layers': self.distill_teacher_ir_layers,
                'distill_student_rgb_channels': self.distill_student_rgb_channels,
                'distill_student_ir_channels': self.distill_student_ir_channels,
                'distill_student_fusion_layers': self.distill_student_fusion_layers,
                'distill_student_fusion_channels': self.distill_student_fusion_channels,
                'distill_teacher_rgb_channels': self.distill_teacher_rgb_channels,
                'distill_teacher_ir_channels': self.distill_teacher_ir_channels,
            }
            for key, value in optional_ddp_keys.items():
                if value is not None:
                    ddp_overrides[key] = value
            if self.Distillation == "MultiDistillation":
                if self.teacher_model_rgb_path is not None:
                    ddp_overrides['Teacher_Model_RGB_Path'] = str(self.teacher_model_rgb_path)
                if self.teacher_model_ir_path is not None:
                    ddp_overrides['Teacher_Model_IR_Path'] = str(self.teacher_model_ir_path)
            elif self.teacher_model_path is not None:
                ddp_overrides['Teacher_Model_Path'] = str(self.teacher_model_path)
            self.ddp_extra_overrides = ddp_overrides
        else:  # 把不需要的args出栈
            distill_args = {
                "Distillation", "loss_type", "online",
                "Teacher_Model_RGB", "Teacher_Model_IR",
                "Teacher_Model", "distill_weight"
            }
            self.Distillation = None
            self.ddp_extra_overrides = {}
            for item in distill_args:
                if item in overrides:
                    overrides.pop(item)

        overrides.update(stage2_overrides)
        overrides.update({
            'distill_disable_all': self.distill_disable_all,
            'distill_cross_attention': self.distill_cross_attention,
            'distill_normal_distillation': self.distill_normal_distillation,
        })
        overrides.update(distill_toggle_overrides)
        overrides.update(audit_overrides)
        self.ddp_extra_overrides.update(distill_toggle_overrides)
        self.ddp_extra_overrides.update(audit_overrides)
        self.args = get_cfg(cfg, overrides)
        if self.distill_stage2_cls_only and self.stage2_disable_early_stop:
            self.args.patience = max(int(getattr(self.args, 'patience', 0) or 0), self.stage2_epochs + 1)
        if self.audit_mode and self.audit_disable_early_stop:
            self.args.patience = max(int(getattr(self.args, 'patience', 0) or 0), int(self.args.epochs) + 1, int(self.args.epochs) * 2)
        if self.audit_mode:
            self.args.val = True
            self.args.save_period = max(int(getattr(self.args, 'save_period', 0) or 0), self.audit_ckpt_interval)
        self.check_resume(overrides)
        self._refresh_custom_override_state_from_args()
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        self.train_augment_target = bool(self.args.augment)
        self.augment_start_epoch = max(int(getattr(self.args, 'augment_start_epoch', 0) or 0), 0)
        self.args.train_augment_active = self.train_augment_target and self.start_epoch >= self.augment_start_epoch
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split('.')[-1] in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]
        if self.audit_mode and RANK in (-1, 0):
            self.audit_logger = TrainingHeadroomAuditLogger(self.save_dir, self.audit_output_name)
            self.audit_probe_fraction_values = self.audit_probe_registry.parse_probe_fractions(self.audit_probe_fractions)
            self.audit_checkpoint_epochs = self._compute_audit_checkpoint_epochs()
            self._sync_audit_checkpoint_index()

        self.FIA = False
        self.pool_for_FIA = None
        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    @staticmethod
    def _safe_float(value, default=float('nan')):
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            value = value.detach().float().mean().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(value) or math.isinf(value):
            return default
        return value

    def _refresh_custom_override_state_from_args(self):
        self.distill_disable_all = bool(getattr(self.args, 'distill_disable_all', self.distill_disable_all))
        self.distill_cross_attention = bool(
            getattr(self.args, 'distill_cross_attention', self.distill_cross_attention))
        self.distill_normal_distillation = bool(
            getattr(self.args, 'distill_normal_distillation', self.distill_normal_distillation))
        self.distill_only_normal = bool(getattr(self.args, 'distill_only_normal', self.distill_only_normal))
        self.distill_disable_gate_kd = bool(
            getattr(self.args, 'distill_disable_gate_kd', self.distill_disable_gate_kd))
        self.distill_disable_pseudo_fusion_kd = bool(
            getattr(self.args, 'distill_disable_pseudo_fusion_kd', self.distill_disable_pseudo_fusion_kd))
        self.distill_head_kd_policy = str(getattr(self.args, 'distill_head_kd_policy', self.distill_head_kd_policy))
        self.distill_epoch_scale_mode = str(
            getattr(self.args, 'distill_epoch_scale_mode', self.distill_epoch_scale_mode))
        self.distill_epoch_scale_start = float(
            getattr(self.args, 'distill_epoch_scale_start', self.distill_epoch_scale_start))
        self.distill_epoch_scale_end = float(
            getattr(self.args, 'distill_epoch_scale_end', self.distill_epoch_scale_end))
        self.distill_epoch_scale_decay_start = int(
            getattr(self.args, 'distill_epoch_scale_decay_start', self.distill_epoch_scale_decay_start))
        self.distill_epoch_scale_decay_end = int(
            getattr(self.args, 'distill_epoch_scale_decay_end', self.distill_epoch_scale_decay_end))
        self.distill_zero_after_epoch = int(
            getattr(self.args, 'distill_zero_after_epoch', self.distill_zero_after_epoch))
        self.distill_schedule_enable = bool(getattr(self.args, 'distill_schedule_enable', self.distill_schedule_enable))
        self.distill_mid_kd_start_epoch = int(
            getattr(self.args, 'distill_mid_kd_start_epoch', self.distill_mid_kd_start_epoch))
        self.distill_mid_kd_policy = str(getattr(self.args, 'distill_mid_kd_policy', self.distill_mid_kd_policy))
        self.distill_student_entropy_weight_enable = bool(getattr(
            self.args, 'distill_student_entropy_weight_enable', self.distill_student_entropy_weight_enable))
        self.distill_student_entropy_weight_mode = str(getattr(
            self.args, 'distill_student_entropy_weight_mode', self.distill_student_entropy_weight_mode))
        self.distill_student_entropy_weight_formula = str(getattr(
            self.args, 'distill_student_entropy_weight_formula', self.distill_student_entropy_weight_formula))
        self.distill_student_entropy_weight_min = float(getattr(
            self.args, 'distill_student_entropy_weight_min', self.distill_student_entropy_weight_min))
        self.distill_student_entropy_weight_max = float(getattr(
            self.args, 'distill_student_entropy_weight_max', self.distill_student_entropy_weight_max))
        self.distill_student_entropy_weight_beta = float(getattr(
            self.args, 'distill_student_entropy_weight_beta', self.distill_student_entropy_weight_beta))
        self.distill_student_entropy_weight_beta_pos = float(getattr(
            self.args, 'distill_student_entropy_weight_beta_pos', self.distill_student_entropy_weight_beta_pos))
        self.distill_student_entropy_weight_beta_neg = float(getattr(
            self.args, 'distill_student_entropy_weight_beta_neg', self.distill_student_entropy_weight_beta_neg))
        self.distill_student_entropy_weight_tau = float(getattr(
            self.args, 'distill_student_entropy_weight_tau', self.distill_student_entropy_weight_tau))
        self.distill_student_entropy_weight_normalize_mean = bool(getattr(
            self.args, 'distill_student_entropy_weight_normalize_mean',
            self.distill_student_entropy_weight_normalize_mean))
        self.distill_student_entropy_weight_detach = bool(getattr(
            self.args, 'distill_student_entropy_weight_detach', self.distill_student_entropy_weight_detach))
        self.distill_student_entropy_weight_target = str(getattr(
            self.args, 'distill_student_entropy_weight_target', self.distill_student_entropy_weight_target))
        self.distill_student_entropy_weight_reg = float(getattr(
            self.args, 'distill_student_entropy_weight_reg', self.distill_student_entropy_weight_reg))
        self.distill_late_kd_start_epoch = int(
            getattr(self.args, 'distill_late_kd_start_epoch', self.distill_late_kd_start_epoch))
        self.distill_late_kd_policy = str(getattr(self.args, 'distill_late_kd_policy', self.distill_late_kd_policy))

    def _compute_distill_epoch_scale(self, epoch):
        """Return the epoch-level teacher scale for KD losses.

        The default legacy_cosine branch is intentionally identical to the previous hard-coded formula.
        New cosine/linear modes use a configurable decay window and may decay all the way to zero.
        """
        epoch = int(epoch)
        zero_after = int(getattr(self, 'distill_zero_after_epoch', -1))
        if zero_after >= 0 and epoch >= zero_after:
            return 0.0

        mode = str(getattr(self, 'distill_epoch_scale_mode', 'legacy_cosine'))
        if mode == 'none':
            return 1.0
        if mode == 'legacy_cosine':
            # Keep the original epoch-to-scale mapping exactly unchanged for backwards compatibility.
            return ((1 - math.cos(epoch * math.pi / self.epochs)) / 2) * (0.1 - 1) + 1

        start_value = float(getattr(self, 'distill_epoch_scale_start', 1.0))
        end_value = float(getattr(self, 'distill_epoch_scale_end', 0.1))
        decay_start = int(getattr(self, 'distill_epoch_scale_decay_start', 0))
        decay_end = int(getattr(self, 'distill_epoch_scale_decay_end', -1))
        if decay_end < 0:
            decay_end = int(self.epochs)

        # Outside the configured decay window we pin to the boundary values so the schedule is stable.
        if epoch <= decay_start:
            progress = 0.0
        elif epoch >= decay_end:
            progress = 1.0
        else:
            span = max(decay_end - decay_start, 1)
            progress = (epoch - decay_start) / span

        if mode == 'cosine':
            # Smoothly fade from start_value to end_value inside the decay window.
            progress = (1 - math.cos(progress * math.pi)) / 2
        elif mode == 'linear':
            pass
        else:
            raise ValueError(f'Unsupported distill_epoch_scale_mode: {mode}')
        return start_value + (end_value - start_value) * progress

    @staticmethod
    def _metric_lookup(metrics, *keys, default=float('nan')):
        if not metrics:
            return default
        for key in keys:
            if key in metrics:
                return BaseTrainer._safe_float(metrics[key], default)
        return default

    def _compute_audit_checkpoint_epochs(self):
        epochs = max(int(self.epochs), 1)
        scheduled = set()
        interval = max(int(self.audit_ckpt_interval), 1)
        for epoch in range(interval, epochs + 1, interval):
            scheduled.add(epoch)
        for fraction in self.audit_probe_fraction_values:
            epoch = min(max(int(round(fraction * epochs)), 1), epochs)
            scheduled.add(epoch)
        scheduled.add(epochs)
        return sorted(scheduled)

    def _sync_audit_checkpoint_index(self):
        if self.audit_logger is None:
            return
        if not self.audit_saved_checkpoints and self.audit_logger.checkpoint_index_path.exists():
            with self.audit_logger.checkpoint_index_path.open('r', encoding='utf-8') as f:
                self.audit_saved_checkpoints = json.load(f)
        self.audit_logger.write_checkpoint_index(self.audit_saved_checkpoints)

    def _reset_audit_epoch_state(self):
        self.audit_epoch_stats = {}
        self.audit_epoch_grad_samples = []

    def _accumulate_audit_stats(self, stats):
        if not self.audit_mode or RANK not in (-1, 0):
            return
        for key, value in stats.items():
            val = self._safe_float(value)
            if math.isnan(val):
                continue
            state = self.audit_epoch_stats.setdefault(key, {'sum': 0.0, 'count': 0})
            state['sum'] += val
            state['count'] += 1

    def _select_audit_gradient_params(self):
        self.audit_grad_params = []
        self.audit_grad_param_name = ''
        if not self.audit_mode:
            return
        top_modules = getattr(de_parallel(self.model), 'model', [])
        candidates = []
        for idx in range(len(top_modules) - 1, -1, -1):
            module = top_modules[idx]
            params = [p for p in module.parameters() if p.requires_grad]
            if not params:
                continue
            module_name = type(module).__name__
            if module_name in ('OBB', 'Detect', 'C2f', 'C3', 'RepC3', 'EntropyOffsetGateFusion', 'Conv'):
                candidates.append((f'model.{idx}.{module_name}', params))
            if len(candidates) >= 2:
                break
        if candidates:
            for name, params in candidates:
                self.audit_grad_param_name = f'{name}+{self.audit_grad_param_name}'.strip('+')
                self.audit_grad_params.extend(params)
        else:
            all_params = [p for p in de_parallel(self.model).parameters() if p.requires_grad]
            self.audit_grad_params = all_params[-4:]
            self.audit_grad_param_name = 'fallback_last_trainable_params'

    @staticmethod
    def _flatten_grads(grad_list, device):
        flat = [g.detach().reshape(-1).float() for g in grad_list if g is not None]
        if not flat:
            return torch.zeros(0, device=device)
        return torch.cat(flat)

    def _compute_audit_gradient_diagnostics(self, det_loss, kd_loss):
        if not self.audit_grad_params or det_loss is None:
            return {}
        try:
            det_grads = torch.autograd.grad(det_loss, self.audit_grad_params, retain_graph=True, allow_unused=True)
            det_flat = self._flatten_grads(det_grads, det_loss.device)
            det_norm = det_flat.norm().item() if det_flat.numel() else 0.0

            if kd_loss is None or self._safe_float(kd_loss, 0.0) == 0.0:
                return {
                    'grad_param_group': self.audit_grad_param_name,
                    'grad_det_norm': det_norm,
                    'grad_kd_norm': 0.0,
                    'grad_cosine': float('nan'),
                }

            kd_grads = torch.autograd.grad(kd_loss, self.audit_grad_params, retain_graph=True, allow_unused=True)
            kd_flat = self._flatten_grads(kd_grads, det_loss.device)
            kd_norm = kd_flat.norm().item() if kd_flat.numel() else 0.0
            if det_flat.numel() and kd_flat.numel() and det_norm > 0 and kd_norm > 0:
                cosine = F.cosine_similarity(det_flat.unsqueeze(0), kd_flat.unsqueeze(0), dim=1).item()
            else:
                cosine = float('nan')
            return {
                'grad_param_group': self.audit_grad_param_name,
                'grad_det_norm': det_norm,
                'grad_kd_norm': kd_norm,
                'grad_cosine': cosine,
            }
        except RuntimeError as exc:
            LOGGER.warning(f'WARNING ⚠️ audit gradient diagnostic skipped: {exc}')
            return {}

    def _audit_record_step(self, batch, ni, loss_items, det_loss_total, det_loss_grad_source=None, distill_grad_source=None):
        if not self.audit_mode or RANK not in (-1, 0):
            return
        criterion_cache = getattr(getattr(de_parallel(self.model), 'criterion', None), 'kd_cache', {}) or {}
        stats = {
            'box_loss': loss_items[0] if len(loss_items) > 0 else float('nan'),
            'cls_loss': loss_items[1] if len(loss_items) > 1 else float('nan'),
            'dfl_loss': loss_items[2] if len(loss_items) > 2 else float('nan'),
            'angle_loss': criterion_cache.get('loss_angle', float('nan')),
            'distill_total_loss': distill_grad_source.detach() if isinstance(distill_grad_source, torch.Tensor) else 0.0,
            'normal_distill_loss': 0.0,
            'cross_distill_loss': 0.0,
            'fg_ratio': criterion_cache.get('fg_mask', torch.zeros((), device=self.device)).float().mean()
            if criterion_cache.get('fg_mask') is not None else 0.0,
            'fg_count': criterion_cache.get('fg_mask', torch.zeros((), device=self.device)).float().sum()
            if criterion_cache.get('fg_mask') is not None else 0.0,
            'batch_instances': batch['cls'].shape[0],
            'det_loss_total': det_loss_total,
        }
        if self.Distillation is not None and getattr(self, 'distillation_loss', None) is not None:
            distill_stats = getattr(self.distillation_loss, 'last_loss_dict', {}) or {}
            stats['normal_distill_loss'] = distill_stats.get('normal_distill_loss', getattr(self, 'd_loss', 0.0))
            stats['cross_distill_loss'] = distill_stats.get('cross_distill_loss', getattr(self, 'c_loss', 0.0))
            for key, value in distill_stats.items():
                stats[f'distill/{key}'] = value
        self._accumulate_audit_stats(stats)

        log_every_n = max(int(self.audit_log_every_n), 1)
        if ni % log_every_n == 0:
            grad_stats = self._compute_audit_gradient_diagnostics(det_loss_grad_source, distill_grad_source)
            if grad_stats:
                self.audit_epoch_grad_samples.append(grad_stats)

    def _build_audit_epoch_record(self):
        if not self.audit_mode or self.audit_logger is None:
            return None

        def avg(key, default=float('nan')):
            state = self.audit_epoch_stats.get(key)
            if not state or state['count'] == 0:
                return default
            return state['sum'] / state['count']

        val_map50 = self._metric_lookup(self.metrics, 'metrics/mAP50(B)', 'metrics/mAP50(OBB)')
        val_map5095 = self._metric_lookup(self.metrics, 'metrics/mAP50-95(B)', 'metrics/mAP50-95(OBB)')
        record = {
            'epoch': int(self.epoch + 1),
            'lr': self._safe_float(next(iter(self.lr.values()), float('nan'))),
            'lr/pg0': self._safe_float(self.lr.get('lr/pg0')),
            'lr/pg1': self._safe_float(self.lr.get('lr/pg1')),
            'lr/pg2': self._safe_float(self.lr.get('lr/pg2')),
            'train/mAP50': float('nan'),
            'train/mAP50-95': float('nan'),
            'val/mAP50': val_map50,
            'val/mAP50-95': val_map5095,
            'box_loss': avg('box_loss'),
            'cls_loss': avg('cls_loss'),
            'dfl_loss': avg('dfl_loss'),
            'angle_loss': avg('angle_loss'),
            'distill_total_loss': avg('distill_total_loss', 0.0),
            'normal_distill_loss': avg('normal_distill_loss', 0.0),
            'cross_distill_loss': avg('cross_distill_loss', 0.0),
            'geometry_gap': val_map50 - val_map5095 if not math.isnan(val_map50) and not math.isnan(val_map5095) else float('nan'),
            'fg_ratio': avg('fg_ratio', 0.0),
            'fg_count': avg('fg_count', 0.0),
            'selected_positive_count': avg('distill/num_selected_pos', 0.0),
            'teacher_reliability_mean': avg('distill/avg_teacher_reliability', 0.0),
            'teacher_reliability_std': avg('distill/std_teacher_reliability', 0.0),
            'teacher_jsd_mean': avg('distill/avg_jsd_on_selected', 0.0),
            'teacher_jsd_std': avg('distill/std_jsd_on_selected', 0.0),
            'gate_rgb_ratio': avg('distill/gate_rgb_ratio', avg('gate_rgb_ratio', 0.0)),
            'gate_ir_ratio': avg('distill/gate_ir_ratio', avg('gate_ir_ratio', 0.0)),
            'gate_entropy': avg('distill/gate_entropy', avg('gate_entropy', 0.0)),
            'align_conf_mean': avg('distill/align_conf_mean', avg('align_conf_mean', 0.0)),
            'distill_gate_kd_enabled': avg('distill/gate_kd_enabled', avg('distill/distill_gate_kd_enabled', 0.0)),
            'distill_pseudo_fusion_enabled': avg(
                'distill/pseudo_fusion_enabled', avg('distill/distill_pseudo_fusion_enabled', 0.0)),
            'distill/schedule_stage_id': avg('distill/schedule_stage_id', 0.0),
            'distill/effective_normal_enabled': avg('distill/effective_normal_enabled', 0.0),
            'distill/cross_attention_enabled': avg('distill/cross_attention_enabled', 0.0),
            'distill/mid_kd_active': avg('distill/mid_kd_active', 0.0),
            'distill/mid_kd_policy_id': avg('distill/mid_kd_policy_id', 0.0),
            'distill/late_kd_active': avg('distill/late_kd_active', 0.0),
            'distill/late_kd_policy_id': avg('distill/late_kd_policy_id', 0.0),
            'distill/head_cls_kd_enabled': avg('distill/head_cls_kd_enabled', 0.0),
            'distill/head_dfl_kd_enabled': avg('distill/head_dfl_kd_enabled', 0.0),
            'distill/head_angle_kd_enabled': avg('distill/head_angle_kd_enabled', 0.0),
            'distill/head_kd_policy_id': avg('distill/head_kd_policy_id', 0.0),
            'distill/head_cls_kd': avg('distill/head_cls_kd', 0.0),
            'distill/head_dfl_kd': avg('distill/head_dfl_kd', 0.0),
            'distill/head_angle_kd': avg('distill/head_angle_kd', 0.0),
            'distill/gate_kd_mode_is_normalized': avg('distill/gate_kd_mode_is_normalized', 0.0),
            'distill/gate_kd_weight': avg('distill/gate_kd_weight', 0.0),
            'distill/gate_kd_conf_thr': avg('distill/gate_kd_conf_thr', 0.0),
            'distill/gate_kd_valid_ratio': avg('distill/gate_kd_valid_ratio', 0.0),
            'distill/gate_kd_raw_normalized': avg('distill/gate_kd_raw_normalized', 0.0),
            'distill/gate_kd': avg('distill/gate_kd', 0.0),
            'distill/student_entropy_mean': avg('distill/student_entropy_mean', 0.0),
            'distill/student_entropy_weight_mean': avg('distill/student_entropy_weight_mean', 0.0),
            'distill/student_entropy_weight_std': avg('distill/student_entropy_weight_std', 0.0),
            'distill/final_kd_weight_mean': avg('distill/final_kd_weight_mean', 0.0),
            'distill/student_entropy_weight_reg_loss': avg('distill/student_entropy_weight_reg_loss', 0.0),
            'distill/distill_epoch_scale': avg('distill/distill_epoch_scale', 1.0),
        }

        for key, state in sorted(self.audit_epoch_stats.items()):
            if key not in record and state['count'] > 0:
                record[key] = state['sum'] / state['count']

        if self.audit_epoch_grad_samples:
            record['grad_param_group'] = self.audit_epoch_grad_samples[-1].get('grad_param_group', self.audit_grad_param_name)
            for key in ('grad_det_norm', 'grad_kd_norm', 'grad_cosine'):
                values = [self._safe_float(sample.get(key)) for sample in self.audit_epoch_grad_samples]
                values = [v for v in values if not math.isnan(v)]
                record[key] = float(sum(values) / len(values)) if values else float('nan')
        else:
            record['grad_param_group'] = self.audit_grad_param_name
            record['grad_det_norm'] = float('nan')
            record['grad_kd_norm'] = float('nan')
            record['grad_cosine'] = float('nan')

        validator_metrics = getattr(getattr(self, 'validator', None), 'metrics', None)
        if validator_metrics is not None:
            maps = getattr(validator_metrics, 'maps', None)
            if maps is not None:
                names = getattr(validator_metrics, 'names', {}) or {}
                per_class_ap = {}
                for idx, value in enumerate(np.asarray(maps).tolist()):
                    per_class_ap[str(names.get(idx, idx))] = float(value)
                record['per_class_ap'] = per_class_ap
        return record

    def _write_audit_epoch_record(self):
        record = self._build_audit_epoch_record()
        if record is not None:
            self.audit_logger.append_epoch_record(record)

    def _save_audit_checkpoint(self, ckpt):
        if not self.audit_mode or self.audit_logger is None:
            return
        epoch_num = int(self.epoch + 1)
        if epoch_num not in self.audit_checkpoint_epochs:
            return
        ckpt_path = self.audit_logger.checkpoints_dir / f'epoch{epoch_num:04d}.pt'
        torch.save(ckpt, ckpt_path)
        record = {
            'epoch': epoch_num,
            'fraction': round(epoch_num / max(int(self.epochs), 1), 6),
            'path': str(ckpt_path),
            'is_probe_source': epoch_num in {
                min(max(int(round(frac * max(int(self.epochs), 1))), 1), max(int(self.epochs), 1))
                for frac in self.audit_probe_fraction_values
            },
        }
        self.audit_saved_checkpoints = [r for r in self.audit_saved_checkpoints if int(r.get('epoch', -1)) != epoch_num]
        self.audit_saved_checkpoints.append(record)
        self.audit_saved_checkpoints.sort(key=lambda item: int(item['epoch']))
        self._sync_audit_checkpoint_index()

    def _infer_teacher_arch(self):
        return 'hgnetv2' if len(self.distill_teacher_rgb_layers or []) == 4 else 'yolov8'

    def _read_audit_checkpoint_records(self):
        if self.audit_logger is None or not self.audit_logger.checkpoint_index_path.exists():
            return []
        with self.audit_logger.checkpoint_index_path.open('r', encoding='utf-8') as f:
            return json.load(f)

    def _resolve_probe_distill_toggles(self, overrides):
        overrides = dict(overrides or {})
        distill_disabled = bool(overrides.get('distill_disable_all', self.distill_disable_all))
        distill_only_normal = bool(overrides.get('distill_only_normal', self.distill_only_normal))
        finegrained_enabled = bool(
            getattr(getattr(self, 'distillation_loss', None), 'enable_finegrained', bool(self.distill_student_fusion_layers))
        )
        schedule_enabled = bool(overrides.get('distill_schedule_enable', self.distill_schedule_enable))
        mid_start_epoch = int(overrides.get('distill_mid_kd_start_epoch', self.distill_mid_kd_start_epoch))
        mid_policy = str(overrides.get('distill_mid_kd_policy', self.distill_mid_kd_policy))
        late_start_epoch = int(overrides.get('distill_late_kd_start_epoch', self.distill_late_kd_start_epoch))
        late_policy = str(overrides.get('distill_late_kd_policy', self.distill_late_kd_policy))
        cross_enabled = bool(overrides.get('distill_cross_attention', self.distill_cross_attention))
        gate_disabled = bool(overrides.get('distill_disable_gate_kd', self.distill_disable_gate_kd))
        pseudo_disabled = bool(overrides.get('distill_disable_pseudo_fusion_kd', self.distill_disable_pseudo_fusion_kd))
        head_policy = str(overrides.get('distill_head_kd_policy', self.distill_head_kd_policy))
        probe_epoch = int(overrides.get('_probe_checkpoint_epoch', -1))
        mid_active = schedule_enabled and mid_start_epoch >= 0 and probe_epoch >= 0 and probe_epoch >= mid_start_epoch
        late_active = late_start_epoch >= 0 and probe_epoch >= 0 and probe_epoch >= late_start_epoch
        if mid_active and not late_active and mid_policy == 'normal_only_gate_off':
            cross_enabled = False
            gate_disabled = True
        if late_active:
            if late_policy == 'gate_off':
                gate_disabled = True
            elif late_policy == 'normal_gate_off':
                cross_enabled = False
                gate_disabled = True
            elif late_policy == 'strict_only_normal':
                cross_enabled = False
                gate_disabled = True
                pseudo_disabled = True
            elif late_policy == 'strict_head_kd_off':
                cross_enabled = False
                gate_disabled = True
                pseudo_disabled = True
                head_policy = 'off'

        gate_enabled = (
            self.Distillation is not None
            and not distill_disabled
            and not distill_only_normal
            and finegrained_enabled
            and not gate_disabled
        )
        pseudo_enabled = (
            self.Distillation is not None
            and not distill_disabled
            and not distill_only_normal
            and finegrained_enabled
            and not pseudo_disabled
        )
        return {
            'distill_gate_kd_enabled': gate_enabled,
            'distill_pseudo_fusion_enabled': pseudo_enabled,
            'distill_cross_attention_enabled': (
                self.Distillation is not None and not distill_disabled and not distill_only_normal and cross_enabled
            ),
            'distill_schedule_stage_id': (
                Multimodal_Distillation_loss.SCHEDULE_STAGE_IDS['late'] if (schedule_enabled and late_active)
                else Multimodal_Distillation_loss.SCHEDULE_STAGE_IDS['mid'] if (schedule_enabled and mid_active and not late_active)
                else Multimodal_Distillation_loss.SCHEDULE_STAGE_IDS['base']
            ),
            'distill_effective_normal_enabled': (
                self.Distillation is not None and not distill_disabled and bool(
                    overrides.get('distill_normal_distillation', self.distill_normal_distillation))
            ),
            'distill_mid_kd_active': schedule_enabled and mid_active and not late_active,
            'distill_mid_kd_policy_id': Multimodal_Distillation_loss.MID_KD_POLICY_IDS.get(mid_policy, 0),
            'distill_late_kd_active': late_active,
            'distill_late_kd_policy_id': Multimodal_Distillation_loss.LATE_KD_POLICY_IDS.get(late_policy, 0),
            'distill_head_kd_policy_id': Multimodal_Distillation_loss.HEAD_KD_POLICY_IDS.get(head_policy, 0),
            'distill_gate_kd_mode': str(overrides.get('distill_gate_kd_mode', self.distill_gate_kd_mode)),
            'distill_gate_kd_weight': float(overrides.get('distill_gate_kd_weight', self.distill_gate_kd_weight)),
            'distill_gate_kd_conf_thr': float(overrides.get('distill_gate_kd_conf_thr', self.distill_gate_kd_conf_thr)),
        }

    def _build_probe_command(self, checkpoint_record, probe_spec):
        script_path = Path(__file__).resolve().parents[2] / 'train_student_deimhgnetv2_obb.py'
        checkpoint_path = Path(checkpoint_record['path'])
        checkpoint_epoch = int(checkpoint_record['epoch'])
        total_epochs_target = checkpoint_epoch + int(self.audit_probe_epochs)
        teacher_arch = self._infer_teacher_arch()
        overrides = dict(probe_spec.get('overrides', {}))
        probe_dir = self.audit_logger.audit_dir / 'probes' / f'epoch{checkpoint_epoch:04d}' / probe_spec['name']
        probe_dir.mkdir(parents=True, exist_ok=True)

        lr0 = float(self.args.lr0)
        if 'lr_scale' in overrides:
            lr0 *= float(overrides.pop('lr_scale'))

        cmd = [
            sys.executable,
            str(script_path),
            '--resume', str(checkpoint_path),
            '--teacher-arch', teacher_arch,
            '--model', str(self.args.model),
            '--data', str(self.args.data),
            '--save-dir', str(probe_dir),
            '--imgsz', str(self.args.imgsz),
            '--epochs', str(total_epochs_target),
            '--patience', str(max(total_epochs_target + 1, total_epochs_target * 2)),
            '--batch', str(self.args.batch),
            '--device', str(self.args.device),
            '--workers', str(self.args.workers),
            '--optimizer', str(self.args.optimizer),
            '--lr0', str(lr0),
            '--loss-type', str(getattr(self, 'loss_type', 'CWD')),
            '--distill-weight', str(float(overrides.pop('distill_weight', getattr(self, 'distill_weight', 0.0)))),
            '--audit-output-name', self.audit_output_name,
            '--audit-eval-split', self.audit_eval_split,
            '--audit-log-every-n', str(int(self.audit_log_every_n)),
            '--audit-ckpt-interval', str(int(self.audit_ckpt_interval)),
            '--audit-mode',
        ]

        if self.teacher_model_rgb_path is not None:
            cmd += ['--teacher-rgb', str(self.teacher_model_rgb_path)]
        if self.teacher_model_ir_path is not None:
            cmd += ['--teacher-ir', str(self.teacher_model_ir_path)]

        bool_flags = {
            'online': bool(getattr(self, 'online', False)),
            'augment': bool(getattr(self.args, 'augment', True)),
            'amp': bool(getattr(self.args, 'amp', False)),
            'rect': bool(getattr(self.args, 'rect', False)),
            'exist-ok': True,
            'distill-cross-attention': bool(overrides.pop('distill_cross_attention', self.distill_cross_attention)),
            'distill-normal-distillation': bool(overrides.pop('distill_normal_distillation', self.distill_normal_distillation)),
            'distill-only-normal': bool(overrides.pop('distill_only_normal', self.distill_only_normal)),
            'distill-disable-gate-kd': bool(overrides.pop('distill_disable_gate_kd', self.distill_disable_gate_kd)),
            'distill-disable-pseudo-fusion-kd': bool(
                overrides.pop('distill_disable_pseudo_fusion_kd', self.distill_disable_pseudo_fusion_kd)),
            'distill-student-entropy-weight-enable': bool(
                overrides.pop('distill_student_entropy_weight_enable', self.distill_student_entropy_weight_enable)),
            'distill-student-entropy-weight-normalize-mean': bool(
                overrides.pop('distill_student_entropy_weight_normalize_mean',
                              self.distill_student_entropy_weight_normalize_mean)),
            'distill-student-entropy-weight-detach': bool(
                overrides.pop('distill_student_entropy_weight_detach', self.distill_student_entropy_weight_detach)),
        }
        distill_disable_all_flag = bool(overrides.pop('distill_disable_all', self.distill_disable_all))
        cmd.append('--distill-disable-all' if distill_disable_all_flag else '--no-distill-disable-all')
        for flag, enabled in bool_flags.items():
            cmd.append(f'--{flag}' if enabled else f'--no-{flag}')

        scalar_flags = {
            'distill-gate-kd-mode': overrides.pop('distill_gate_kd_mode', self.distill_gate_kd_mode),
            'distill-gate-kd-weight': overrides.pop('distill_gate_kd_weight', self.distill_gate_kd_weight),
            'distill-gate-kd-temperature': overrides.pop('distill_gate_kd_temperature', self.distill_gate_kd_temperature),
            'distill-gate-kd-mask-mode': overrides.pop('distill_gate_kd_mask_mode', self.distill_gate_kd_mask_mode),
            'distill-gate-kd-conf-thr': overrides.pop('distill_gate_kd_conf_thr', self.distill_gate_kd_conf_thr),
            'distill-student-entropy-weight-mode': overrides.pop(
                'distill_student_entropy_weight_mode', self.distill_student_entropy_weight_mode),
            'distill-student-entropy-weight-formula': overrides.pop(
                'distill_student_entropy_weight_formula', self.distill_student_entropy_weight_formula),
            'distill-student-entropy-weight-min': overrides.pop(
                'distill_student_entropy_weight_min', self.distill_student_entropy_weight_min),
            'distill-student-entropy-weight-max': overrides.pop(
                'distill_student_entropy_weight_max', self.distill_student_entropy_weight_max),
            'distill-student-entropy-weight-beta': overrides.pop(
                'distill_student_entropy_weight_beta', self.distill_student_entropy_weight_beta),
            'distill-student-entropy-weight-beta-pos': overrides.pop(
                'distill_student_entropy_weight_beta_pos', self.distill_student_entropy_weight_beta_pos),
            'distill-student-entropy-weight-beta-neg': overrides.pop(
                'distill_student_entropy_weight_beta_neg', self.distill_student_entropy_weight_beta_neg),
            'distill-student-entropy-weight-tau': overrides.pop(
                'distill_student_entropy_weight_tau', self.distill_student_entropy_weight_tau),
            'distill-student-entropy-weight-target': overrides.pop(
                'distill_student_entropy_weight_target', self.distill_student_entropy_weight_target),
            'distill-student-entropy-weight-reg': overrides.pop(
                'distill_student_entropy_weight_reg', self.distill_student_entropy_weight_reg),
            'distill-head-kd-policy': overrides.pop('distill_head_kd_policy', self.distill_head_kd_policy),
            'distill-late-kd-start-epoch': overrides.pop('distill_late_kd_start_epoch', self.distill_late_kd_start_epoch),
            'distill-late-kd-policy': overrides.pop('distill_late_kd_policy', self.distill_late_kd_policy),
            'distill-epoch-scale-mode': overrides.pop('distill_epoch_scale_mode', self.distill_epoch_scale_mode),
            'distill-epoch-scale-start': overrides.pop('distill_epoch_scale_start', self.distill_epoch_scale_start),
            'distill-epoch-scale-end': overrides.pop('distill_epoch_scale_end', self.distill_epoch_scale_end),
            'distill-epoch-scale-decay-start': overrides.pop(
                'distill_epoch_scale_decay_start', self.distill_epoch_scale_decay_start),
            'distill-epoch-scale-decay-end': overrides.pop(
                'distill_epoch_scale_decay_end', self.distill_epoch_scale_decay_end),
            'distill-zero-after-epoch': overrides.pop('distill_zero_after_epoch', self.distill_zero_after_epoch),
        }
        for flag, value in scalar_flags.items():
            cmd += [f'--{flag}', str(value)]

        if overrides.pop('distill_stage2_cls_only', False):
            cmd.append('--distill-stage2-cls-only')
            cmd += ['--distill-resume-ckpt', str(overrides.pop('distill_resume_ckpt', checkpoint_path))]
            cmd += ['--stage2-epochs', str(total_epochs_target if overrides.pop('use_probe_epochs_as_stage2_epochs', False)
                                           else int(self.stage2_epochs))]
            for flag, enabled in {
                'freeze-backbone-fusion': bool(overrides.pop('freeze_backbone_fusion', self.freeze_backbone_fusion)),
                'train-neck-head-only': bool(overrides.pop('train_neck_head_only', self.train_neck_head_only)),
                'stage2-disable-early-stop': bool(overrides.pop('stage2_disable_early_stop', True)),
            }.items():
                cmd.append(f'--{flag}' if enabled else f'--no-{flag}')

        return cmd, probe_dir, total_epochs_target

    def _read_probe_metrics_from_results(self, save_dir):
        save_dir = Path(save_dir)
        results_path = save_dir / 'results.csv'
        audit_metrics_path = save_dir / self.audit_output_name / 'audit_metrics.csv'
        if not results_path.exists():
            candidates = sorted(save_dir.glob('**/results.csv'))
            if candidates:
                results_path = candidates[0]
        if not audit_metrics_path.exists():
            audit_candidates = sorted(save_dir.glob(f'**/{self.audit_output_name}/audit_metrics.csv'))
            if audit_candidates:
                audit_metrics_path = audit_candidates[0]

        def _load_last_audit_metrics(path):
            defaults = {
                'audit_metrics_csv': None,
                'distill/gate_kd_enabled': float('nan'),
                'distill/pseudo_fusion_enabled': float('nan'),
                'distill/schedule_stage_id': float('nan'),
                'distill/effective_normal_enabled': float('nan'),
                'distill/gate_kd_mode_is_normalized': float('nan'),
                'distill/gate_kd_weight': float('nan'),
                'distill/gate_kd_conf_thr': float('nan'),
                'distill/gate_kd_valid_ratio': float('nan'),
                'distill/gate_kd_raw_normalized': float('nan'),
                'distill/gate_kd': float('nan'),
                'distill/cross_attention_enabled': float('nan'),
                'distill/mid_kd_active': float('nan'),
                'distill/mid_kd_policy_id': float('nan'),
                'distill/late_kd_active': float('nan'),
                'distill/late_kd_policy_id': float('nan'),
                'distill_gate_kd_enabled': float('nan'),
                'distill_pseudo_fusion_enabled': float('nan'),
            }
            if not path.exists():
                return defaults
            with path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                return defaults
            last = rows[-1]
            defaults['audit_metrics_csv'] = str(path)
            for key in list(defaults.keys()):
                if key == 'audit_metrics_csv':
                    continue
                if key in last and last[key] not in ('', None):
                    defaults[key] = self._safe_float(last[key])
            return defaults

        if not results_path.exists():
            return {
                'best_val_mAP50': float('nan'),
                'best_val_mAP50_95': float('nan'),
                'best_epoch': None,
                'results_csv': None,
                **_load_last_audit_metrics(audit_metrics_path),
            }
        with results_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = [{(k or '').strip(): v for k, v in row.items()} for row in reader]
        if not rows:
            return {
                'best_val_mAP50': float('nan'),
                'best_val_mAP50_95': float('nan'),
                'best_epoch': None,
                'results_csv': str(results_path),
                **_load_last_audit_metrics(audit_metrics_path),
            }
        def _find_value(row, *keys):
            for key in keys:
                if key in row and row[key] not in ('', None):
                    return self._safe_float(row[key])
            return float('nan')
        best_map5095 = float('-inf')
        best_map50 = float('nan')
        best_epoch = None
        for row in rows:
            map5095 = _find_value(row, 'metrics/mAP50-95(B)', 'metrics/mAP50-95(OBB)')
            if math.isnan(map5095):
                continue
            if map5095 > best_map5095:
                best_map5095 = map5095
                best_map50 = _find_value(row, 'metrics/mAP50(B)', 'metrics/mAP50(OBB)')
                best_epoch = int(self._safe_float(row.get('epoch'), float('nan'))) if row.get('epoch') else None
        if best_map5095 == float('-inf'):
            best_map5095 = float('nan')
        return {
            'best_val_mAP50': best_map50,
            'best_val_mAP50_95': best_map5095,
            'best_epoch': best_epoch,
            'results_csv': str(results_path),
            **_load_last_audit_metrics(audit_metrics_path),
        }

    def run_audit_probes(self):
        if not self.audit_mode or os.environ.get('SEMD_AUDIT_PROBE_CHILD'):
            return []
        if self.audit_logger is None:
            self.audit_logger = TrainingHeadroomAuditLogger(self.save_dir, self.audit_output_name)

        checkpoint_records = self._read_audit_checkpoint_records()
        if not checkpoint_records:
            LOGGER.warning('WARNING ⚠️ audit mode did not find any saved probe checkpoints to branch from.')
            return []

        context = {
            'supports_head_cls_only': bool(self.distill_resume_ckpt or self.distill_stage2_cls_only or hasattr(self, 'stage2_epochs')),
            'supports_gate_kd_toggle': True,
            'supports_pseudo_fusion_kd_toggle': True,
        }
        available_probes, skipped_probes = self.audit_probe_registry.expand(context, self.audit_probes)
        manifest = {
            'save_dir': str(self.save_dir),
            'audit_dir': str(self.audit_logger.audit_dir),
            'probe_epochs': int(self.audit_probe_epochs),
            'checkpoint_records': checkpoint_records,
            'available_probes': available_probes,
            'skipped_probes': skipped_probes,
            'commands': [],
        }

        probe_results = []
        for checkpoint_record in checkpoint_records:
            if not checkpoint_record.get('is_probe_source', False):
                continue
            ckpt_context = {
                **context,
                'checkpoint_path': checkpoint_record['path'],
            }
            probe_specs, skipped = self.audit_probe_registry.expand(ckpt_context, self.audit_probes)
            if skipped:
                manifest.setdefault('skipped_per_checkpoint', []).append({
                    'epoch': checkpoint_record['epoch'],
                    'skipped': skipped,
                })
                for skipped_probe in skipped:
                    probe_results.append({
                        'checkpoint_epoch': int(checkpoint_record['epoch']),
                        'probe': skipped_probe['name'],
                        'save_dir': '',
                        'returncode': None,
                        'best_val_mAP50': float('nan'),
                        'best_val_mAP50_95': float('nan'),
                        'best_epoch': None,
                        'probe_executed': False,
                        'probe_skipped_reason': skipped_probe.get('reason', 'unknown'),
                        'distill_gate_kd_enabled': None,
                        'distill_pseudo_fusion_enabled': None,
                        'distill_cross_attention_enabled': None,
                        'distill_late_kd_active': None,
                        'distill_late_kd_policy_id': None,
                    })
            for probe_spec in probe_specs:
                cmd, probe_dir, total_epochs_target = self._build_probe_command(checkpoint_record, probe_spec)
                effective_flags = self._resolve_probe_distill_toggles({
                    **probe_spec.get('overrides', {}),
                    '_probe_checkpoint_epoch': checkpoint_record['epoch'],
                })
                manifest['commands'].append({
                    'epoch': checkpoint_record['epoch'],
                    'probe': probe_spec['name'],
                    'save_dir': str(probe_dir),
                    'epochs_target': total_epochs_target,
                    'probe_executed': True,
                    'probe_skipped_reason': '',
                    **effective_flags,
                    'cmd': cmd,
                })
        self.audit_logger.write_probe_manifest(manifest)

        for item in manifest['commands']:
            env = os.environ.copy()
            env['SEMD_AUDIT_PROBE_CHILD'] = '1'
            LOGGER.info(f"[audit] launching probe {item['probe']} from epoch {item['epoch']}")
            proc = subprocess.run(item['cmd'], env=env, check=False)
            metrics = self._read_probe_metrics_from_results(item['save_dir'])
            probe_results.append({
                'checkpoint_epoch': int(item['epoch']),
                'probe': item['probe'],
                'save_dir': item['save_dir'],
                'returncode': int(proc.returncode),
                'probe_executed': True,
                'probe_skipped_reason': '',
                'distill_gate_kd_enabled': item.get('distill_gate_kd_enabled'),
                'distill_pseudo_fusion_enabled': item.get('distill_pseudo_fusion_enabled'),
                'distill_cross_attention_enabled': item.get('distill_cross_attention_enabled'),
                'distill_late_kd_active': item.get('distill_late_kd_active'),
                'distill_late_kd_policy_id': item.get('distill_late_kd_policy_id'),
                'distill_gate_kd_mode': item.get('distill_gate_kd_mode'),
                'distill_gate_kd_weight': item.get('distill_gate_kd_weight'),
                'distill_gate_kd_conf_thr': item.get('distill_gate_kd_conf_thr'),
                **metrics,
            })
        self.audit_logger.write_probe_results(probe_results)
        return probe_results

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(','))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch == -1:
                LOGGER.warning("WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                               "default 'batch=16'")
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device('cuda', RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  # set to enforce timeout
        dist.init_process_group(
            'nccl' if dist.is_nccl_available() else 'gloo',
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size)

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        self.FIA = False

        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True
        self._apply_stage2_freeze()

        # 是否蒸馏
        if self.Distillation is not None:
            """
            self.Distillation : 教师模型

            v.requires_grad = False 离线蒸馏

            v.requires_grad = True  在线蒸馏

            """

            self.__hidden__ = torch.nn.Linear(1, 1, bias=False)
            if self.Distillation == "MultiDistillation":  # 多模态蒸馏
                for k, v in self.Teacher_Model_IR.model.named_parameters():
                    v.requires_grad = self.online
                self.Teacher_Model_IR = self.Teacher_Model_IR.to(self.device)
                for k, v in self.Teacher_Model_RGB.model.named_parameters():
                    v.requires_grad = self.online
                self.Teacher_Model_RGB = self.Teacher_Model_RGB.to(self.device)
            else:  # 单模态蒸馏
                for k, v in self.Teacher_Model.model.named_parameters():
                    v.requires_grad = self.online
                self.Teacher_Model = self.Teacher_Model.to(self.device)

        else:  # 不蒸馏
            self.distillation_loss = None

        # self.set_model_attributes()

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])

            # 是否蒸馏
            if self.Distillation is not None:
                if self.Distillation == "MultiDistillation":  # 多模态蒸馏
                    if self.online:
                        self.Teacher_Model_IR = nn.parallel.DistributedDataParallel(self.Teacher_Model_IR,
                                                                                    device_ids=[RANK])
                        self.Teacher_Model_RGB = nn.parallel.DistributedDataParallel(self.Teacher_Model_RGB,
                                                                                     device_ids=[RANK])
                    self.Teacher_Model_IR.eval()
                    self.Teacher_Model_RGB.eval()
                else:  # 单模态蒸馏
                    if self.online:
                        self.Teacher_Model = nn.parallel.DistributedDataParallel(self.Teacher_Model, device_ids=[RANK])
                    self.Teacher_Model.eval()

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multi-scale training

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        self.args.train_augment_active = self.train_augment_target and self.start_epoch >= self.augment_start_epoch
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            # NOTE: When training DOTA dataset, double batch size could get OOM cause some images got more than 2000 objects.
            self.test_loader = self.get_dataloader(self.testset,
                                                   batch_size=batch_size if self.args.task == 'obb' else batch_size * 2,
                                                   rank=-1,
                                                   mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # 是否蒸馏
        if self.Distillation is not None:
            if self.Distillation == "MultiDistillation":  # 多模态蒸馏
                self.distillation_loss = Multimodal_Distillation_loss(
                    self.model,
                    self.Teacher_Model_RGB,
                    self.Teacher_Model_IR,
                    distiller=self.loss_type,
                    stage2_cls_only=self.distill_stage2_cls_only,
                    distill_cls_kd_weight=self.distill_cls_kd_weight,
                    distill_kd_temperature=self.distill_kd_temperature,
                    teacher_conf_thr=self.teacher_conf_thr,
                    teacher_entropy_thr=self.teacher_entropy_thr,
                    teacher_jsd_thr=self.teacher_jsd_thr,
                    distill_only_normal=self.distill_only_normal,
                    distill_disable_gate_kd=self.distill_disable_gate_kd,
                    distill_disable_pseudo_fusion_kd=self.distill_disable_pseudo_fusion_kd,
                    distill_head_kd_policy=self.distill_head_kd_policy,
                    distill_schedule_enable=self.distill_schedule_enable,
                    distill_mid_kd_start_epoch=self.distill_mid_kd_start_epoch,
                    distill_mid_kd_policy=self.distill_mid_kd_policy,
                    distill_gate_kd_mode=self.distill_gate_kd_mode,
                    distill_gate_kd_weight=self.distill_gate_kd_weight,
                    distill_gate_kd_temperature=self.distill_gate_kd_temperature,
                    distill_gate_kd_mask_mode=self.distill_gate_kd_mask_mode,
                    distill_gate_kd_conf_thr=self.distill_gate_kd_conf_thr,
                    distill_student_entropy_weight_enable=self.distill_student_entropy_weight_enable,
                    distill_student_entropy_weight_mode=self.distill_student_entropy_weight_mode,
                    distill_student_entropy_weight_formula=self.distill_student_entropy_weight_formula,
                    distill_student_entropy_weight_min=self.distill_student_entropy_weight_min,
                    distill_student_entropy_weight_max=self.distill_student_entropy_weight_max,
                    distill_student_entropy_weight_beta=self.distill_student_entropy_weight_beta,
                    distill_student_entropy_weight_beta_pos=self.distill_student_entropy_weight_beta_pos,
                    distill_student_entropy_weight_beta_neg=self.distill_student_entropy_weight_beta_neg,
                    distill_student_entropy_weight_tau=self.distill_student_entropy_weight_tau,
                    distill_student_entropy_weight_normalize_mean=self.distill_student_entropy_weight_normalize_mean,
                    distill_student_entropy_weight_detach=self.distill_student_entropy_weight_detach,
                    distill_student_entropy_weight_target=self.distill_student_entropy_weight_target,
                    distill_student_entropy_weight_reg=self.distill_student_entropy_weight_reg,
                    distill_late_kd_start_epoch=self.distill_late_kd_start_epoch,
                    distill_late_kd_policy=self.distill_late_kd_policy,
                    student_rgb_layer_ids=self.distill_student_rgb_layers,
                    student_ir_layer_ids=self.distill_student_ir_layers,
                    teacher_rgb_layer_ids=self.distill_teacher_rgb_layers,
                    teacher_ir_layer_ids=self.distill_teacher_ir_layers,
                    student_rgb_channels=self.distill_student_rgb_channels,
                    student_ir_channels=self.distill_student_ir_channels,
                    student_fusion_layer_ids=self.distill_student_fusion_layers,
                    student_fusion_channels=self.distill_student_fusion_channels,
                    teacher_rgb_channels=self.distill_teacher_rgb_channels,
                    teacher_ir_channels=self.distill_teacher_ir_channels,
                    cross_attention=self.distill_cross_attention,
                    normal_distillation=self.distill_normal_distillation).to(self.device)
            else:  # 单模态蒸馏
                self.distillation_loss = Distillation_loss(self.model, self.Teacher_Model, distiller=self.loss_type).to(
                    self.device)

        self._select_audit_gradient_params()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs

        if self.Distillation == "MultiDistillation":
            self.optimizer = self.build_optimizer_multi(model=self.model,
                                                        teacher_model_ir=self.Teacher_Model_IR,
                                                        teacher_model_rgb=self.Teacher_Model_RGB,
                                                        distill_loss=self.distillation_loss,
                                                        name=self.args.optimizer,
                                                        lr=self.args.lr0,
                                                        momentum=self.args.momentum,
                                                        decay=weight_decay,
                                                        iterations=iterations)
        else:
            self.optimizer = self.build_optimizer(model=self.model,
                                                  teacher_model=self.Distillation,
                                                  distill_loss=self.distillation_loss,
                                                  name=self.args.optimizer,
                                                  lr=self.args.lr0,
                                                  momentum=self.args.momentum,
                                                  decay=weight_decay,
                                                  iterations=iterations)
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        # 是否蒸馏
        if self.Distillation is not None:
            if self.Distillation == "MultiDistillation":  # 多模态蒸馏
                self.distillation_loss = Multimodal_Distillation_loss(
                    self.model,
                    self.Teacher_Model_RGB,
                    self.Teacher_Model_IR,
                    distiller=self.loss_type,
                    stage2_cls_only=self.distill_stage2_cls_only,
                    distill_cls_kd_weight=self.distill_cls_kd_weight,
                    distill_kd_temperature=self.distill_kd_temperature,
                    teacher_conf_thr=self.teacher_conf_thr,
                    teacher_entropy_thr=self.teacher_entropy_thr,
                    teacher_jsd_thr=self.teacher_jsd_thr,
                    distill_only_normal=self.distill_only_normal,
                    distill_disable_gate_kd=self.distill_disable_gate_kd,
                    distill_disable_pseudo_fusion_kd=self.distill_disable_pseudo_fusion_kd,
                    distill_head_kd_policy=self.distill_head_kd_policy,
                    distill_schedule_enable=self.distill_schedule_enable,
                    distill_mid_kd_start_epoch=self.distill_mid_kd_start_epoch,
                    distill_mid_kd_policy=self.distill_mid_kd_policy,
                    distill_gate_kd_mode=self.distill_gate_kd_mode,
                    distill_gate_kd_weight=self.distill_gate_kd_weight,
                    distill_gate_kd_temperature=self.distill_gate_kd_temperature,
                    distill_gate_kd_mask_mode=self.distill_gate_kd_mask_mode,
                    distill_gate_kd_conf_thr=self.distill_gate_kd_conf_thr,
                    distill_student_entropy_weight_enable=self.distill_student_entropy_weight_enable,
                    distill_student_entropy_weight_mode=self.distill_student_entropy_weight_mode,
                    distill_student_entropy_weight_formula=self.distill_student_entropy_weight_formula,
                    distill_student_entropy_weight_min=self.distill_student_entropy_weight_min,
                    distill_student_entropy_weight_max=self.distill_student_entropy_weight_max,
                    distill_student_entropy_weight_beta=self.distill_student_entropy_weight_beta,
                    distill_student_entropy_weight_beta_pos=self.distill_student_entropy_weight_beta_pos,
                    distill_student_entropy_weight_beta_neg=self.distill_student_entropy_weight_beta_neg,
                    distill_student_entropy_weight_tau=self.distill_student_entropy_weight_tau,
                    distill_student_entropy_weight_normalize_mean=self.distill_student_entropy_weight_normalize_mean,
                    distill_student_entropy_weight_detach=self.distill_student_entropy_weight_detach,
                    distill_student_entropy_weight_target=self.distill_student_entropy_weight_target,
                    distill_student_entropy_weight_reg=self.distill_student_entropy_weight_reg,
                    distill_late_kd_start_epoch=self.distill_late_kd_start_epoch,
                    distill_late_kd_policy=self.distill_late_kd_policy,
                    student_rgb_layer_ids=self.distill_student_rgb_layers,
                    student_ir_layer_ids=self.distill_student_ir_layers,
                    teacher_rgb_layer_ids=self.distill_teacher_rgb_layers,
                    teacher_ir_layer_ids=self.distill_teacher_ir_layers,
                    student_rgb_channels=self.distill_student_rgb_channels,
                    student_ir_channels=self.distill_student_ir_channels,
                    student_fusion_layer_ids=self.distill_student_fusion_layers,
                    student_fusion_channels=self.distill_student_fusion_channels,
                    teacher_rgb_channels=self.distill_teacher_rgb_channels,
                    teacher_ir_channels=self.distill_teacher_ir_channels,
                    cross_attention=self.distill_cross_attention,
                    normal_distillation=self.distill_normal_distillation).to(self.device)
            else:
                self.distillation_loss = Distillation_loss(self.model, self.Teacher_Model, distiller=self.loss_type).to(
                    self.device)

        ##################################### distillation #####################################
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        if self.Distillation == "MultiDistillation":
            self.optimizer = self.build_optimizer_multi(model=self.model,
                                                        teacher_model_ir=self.Teacher_Model_IR,
                                                        teacher_model_rgb=self.Teacher_Model_RGB,
                                                        distill_loss=self.distillation_loss,
                                                        name=self.args.optimizer,
                                                        lr=self.args.lr0,
                                                        momentum=self.args.momentum,
                                                        decay=weight_decay,
                                                        iterations=iterations)
        else:
            self.optimizer = self.build_optimizer(model=self.model,
                                                  teacher_model=self.Distillation,
                                                  distill_loss=self.distillation_loss,
                                                  name=self.args.optimizer,
                                                  lr=self.args.lr0,
                                                  momentum=self.args.momentum,
                                                  decay=weight_decay,
                                                  iterations=iterations)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for '
                    f'{self.args.time} hours...' if self.args.time else f'{self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self._reset_audit_epoch_state()
            self.total_mecd1 = torch.zeros(1, device=self.device)
            self.total_mecd2 = torch.zeros(1, device=self.device)
            self.total_mrl_1 = torch.zeros(1, device=self.device)
            self.total_mrl_2 = torch.zeros(1, device=self.device)
            self.total_FIA = torch.zeros(1, device=self.device)
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            self._enforce_stage2_frozen_modules_eval()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if (not self.args.train_augment_active and self.train_augment_target and
                    epoch >= self.augment_start_epoch):
                self._set_train_augmentation(
                    True,
                    reason=f'epoch {epoch + 1}/{self.epochs}')
                pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()

            # 是否蒸馏
            if self.Distillation is not None:
                self.distillation_loss.register_hook()

            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup

                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    det_loss_grad_source = self.loss.clone()
                    distill_grad_source = None
                    det_loss_total = self.loss_items.sum().detach()
                    if RANK != -1:
                        self.loss *= world_size
                    if self.FIA:
                        RGB_img = batch['img'][:, :3, :, :]
                        # # HSV
                        img_v, _ = torch.max(RGB_img, dim=1, keepdim=True)
                        img_v = img_v.detach()
                        gt = self.pool_for_FIA(img_v)

                        # YCbCr
                        # r = RGB_img[:, 0, :, :]
                        # g = RGB_img[:, 1, :, :]
                        # b = RGB_img[:, 2, :, :]
                        # y = 0.299 * r + 0.587 * g + 0.114 * b
                        # gt = self.pool_for_FIA(y)

                        FIA_module = de_parallel(self.model).model[2]
                        weight = FIA_module(RGB_img)
                        illumination_loss = torch.abs(gt - weight).mean()  # no_sup
                        illumination_loss *= 1.3  # weight
                        self.loss += illumination_loss
                        self.loss_items = torch.cat(
                            (
                                self.loss_items, illumination_loss.unsqueeze(0).detach()),
                            dim=0)
                    # 是否蒸馏
                    if self.Distillation is not None:
                        distill_epoch_scale = self._compute_distill_epoch_scale(self.epoch)
                        if self.Distillation == "MultiDistillation":
                            with torch.no_grad():
                                RGB_img = batch['img'][:, :3, :, :]
                                IR_img = batch['img'][:, 3:, :, :]
                                pred_rgb = self.Teacher_Model_RGB(RGB_img)
                                pred_ir = self.Teacher_Model_IR(IR_img)
                        else:
                            with torch.no_grad():
                                pred = self.Teacher_Model(batch['img'])
                        if self.Distillation == "MultiDistillation":
                            self.d_loss, self.c_loss = self.distillation_loss.get_loss(
                                teacher_preds_rgb=pred_rgb,
                                teacher_preds_ir=pred_ir,
                                batch=batch,
                                current_epoch=self.epoch)
                        else:
                            self.d_loss, self.c_loss = self.distillation_loss.get_loss()
                        if self.distill_stage2_cls_only:
                            cls_kd_weight = self.distill_cls_kd_weight
                            self.d_loss = self.d_loss * cls_kd_weight
                            self.c_loss = self.c_loss * 0.0
                            # Stage-2 cls-only keeps its existing KD behavior and only logs a fixed scale.
                            distill_epoch_scale_for_log = 1.0
                        else:
                            distill_epoch_scale_for_log = distill_epoch_scale
                            # Apply the epoch-level KD scale to all distillation branches together.
                            self.d_loss = self.d_loss * (
                                    self.distill_weight * distill_epoch_scale)
                            self.c_loss = self.c_loss * (
                                    self.distill_weight * distill_epoch_scale)
                        if getattr(self, 'distillation_loss', None) is not None:
                            self.distillation_loss.last_loss_dict['distill_epoch_scale'] = distill_epoch_scale_for_log
                        distill_grad_source = self.d_loss + self.c_loss
                        self.loss += self.d_loss
                        self.loss += self.c_loss
                        if self.distill_stage2_cls_only:
                            stage2_stats = self.distillation_loss.last_loss_dict
                            student_kd_cache = self.distillation_loss._get_student_criterion_cache()
                            det_loss_for_log = student_kd_cache.get('det_loss_total', det_loss_total)
                            stage2_log_tensors = torch.tensor(
                                [
                                    float(det_loss_for_log.detach().cpu()),
                                    float(stage2_stats.get('cls_kd_loss', 0.0)),
                                    float(stage2_stats.get('num_selected_pos', 0.0)),
                                    float(stage2_stats.get('rgb_selected_ratio', 0.0)),
                                    float(stage2_stats.get('ir_selected_ratio', 0.0)),
                                    float(stage2_stats.get('avg_teacher_reliability', 0.0)),
                                    float(stage2_stats.get('avg_jsd_on_selected', 0.0)),
                                ],
                                device=self.device,
                            )
                            self.loss_items = torch.cat((self.loss_items, stage2_log_tensors), dim=0)
                        else:
                            self.loss_items = torch.cat(
                                (self.loss_items, self.d_loss.unsqueeze(0).detach(), self.c_loss.unsqueeze(0).detach()),
                                dim=0)
                    self._audit_record_step(
                        batch=batch,
                        ni=ni,
                        loss_items=self.loss_items,
                        det_loss_total=det_loss_total,
                        det_loss_grad_source=det_loss_grad_source,
                        distill_grad_source=distill_grad_source,
                    )

                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items
                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                losses_list = losses.tolist()
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + len(losses_list))) %
                        (
                            f'{epoch + 1}/{self.epochs}', mem, *losses_list, batch['cls'].shape[0],
                            batch['img'].shape[-1]))
                    # pbar.set_description(
                    #     ('%11s' * 2 + '%11.4g' * (2 + len(losses_list))) %
                    #     (
                    #         f'{epoch + 1}/{self.epochs}', mem, *losses_list, batch['cls'].shape[0],
                    #         self.args.imgsz))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            # 是否蒸馏
            if self.Distillation is not None:
                self.distillation_loss.remove_handle_()

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks('on_train_epoch_end')
            if RANK in (-1, 0):
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self._write_audit_epoch_record()
                self.stop |= self.stopper(epoch + 1, self.fitness)
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks('on_model_save')

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.scheduler.step()
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    @staticmethod
    def _detach_runtime_tensors_for_checkpoint(value):
        """Detach non-leaf runtime tensors so deepcopy-based checkpointing is safe."""
        if isinstance(value, torch.Tensor):
            return value.detach() if not value.is_leaf else value
        if isinstance(value, dict):
            return {k: BaseTrainer._detach_runtime_tensors_for_checkpoint(v) for k, v in value.items()}
        if isinstance(value, list):
            return [BaseTrainer._detach_runtime_tensors_for_checkpoint(v) for v in value]
        if isinstance(value, tuple):
            return tuple(BaseTrainer._detach_runtime_tensors_for_checkpoint(v) for v in value)
        return value

    @staticmethod
    def _clear_object_runtime_caches(obj, seen=None):
        """Clear tensor caches stored on non-module helper objects, e.g. losses/assigners."""
        if obj is None:
            return
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if hasattr(obj, 'kd_cache'):
            obj.kd_cache = {}

        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if hasattr(value, 'kd_cache'):
                    value.kd_cache = {}
                obj[key] = BaseTrainer._detach_runtime_tensors_for_checkpoint(value)
            return

        if isinstance(obj, (list, tuple)):
            for value in obj:
                if hasattr(value, 'kd_cache'):
                    value.kd_cache = {}
            return

        # Loss objects are not always nn.Module instances, so their internal runtime caches would otherwise survive
        # model.modules() traversal and break deepcopy-based checkpointing after the first forward pass.
        if not isinstance(obj, nn.Module) and hasattr(obj, '__dict__'):
            for name, value in list(vars(obj).items()):
                if name.startswith('_'):
                    continue
                if hasattr(value, 'kd_cache'):
                    value.kd_cache = {}
                sanitized = BaseTrainer._detach_runtime_tensors_for_checkpoint(value)
                if sanitized is not value:
                    setattr(obj, name, sanitized)

    @staticmethod
    def _clear_model_runtime_caches(model):
        """Remove transient forward caches before serializing a model checkpoint."""
        if model is None:
            return None
        model = de_parallel(model)
        BaseTrainer._clear_object_runtime_caches(getattr(model, 'criterion', None))
        for module in model.modules():
            if hasattr(module, 'kd_cache'):
                module.kd_cache = {}
            for name, value in list(vars(module).items()):
                if name.startswith('_'):
                    continue
                BaseTrainer._clear_object_runtime_caches(value)
                sanitized = BaseTrainer._detach_runtime_tensors_for_checkpoint(value)
                if sanitized is not value:
                    setattr(module, name, sanitized)
        return model

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import pandas as pd  # scope for faster startup
        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
        train_args = vars(self.args).copy()
        train_args.update({
            'audit_mode': self.audit_mode,
            'audit_disable_early_stop': self.audit_disable_early_stop,
            'audit_ckpt_interval': self.audit_ckpt_interval,
            'audit_probe_epochs': self.audit_probe_epochs,
            'audit_probe_fractions': self.audit_probe_fractions,
            'audit_probes': self.audit_probes,
            'audit_log_every_n': self.audit_log_every_n,
            'audit_output_name': self.audit_output_name,
            'distill_disable_all': self.distill_disable_all,
            'distill_disable_gate_kd': self.distill_disable_gate_kd,
            'distill_disable_pseudo_fusion_kd': self.distill_disable_pseudo_fusion_kd,
        })
        if self.Distillation is not None:
            train_args.update({
                'Distillation': self.Distillation,
                'loss_type': self.loss_type,
                'distill_weight': self.distill_weight,
                'online': self.online,
                'distill_cross_attention': self.distill_cross_attention,
                'distill_normal_distillation': self.distill_normal_distillation,
                'distill_epoch_scale_mode': self.distill_epoch_scale_mode,
                'distill_epoch_scale_start': self.distill_epoch_scale_start,
                'distill_epoch_scale_end': self.distill_epoch_scale_end,
                'distill_epoch_scale_decay_start': self.distill_epoch_scale_decay_start,
                'distill_epoch_scale_decay_end': self.distill_epoch_scale_decay_end,
                'distill_zero_after_epoch': self.distill_zero_after_epoch,
                'distill_student_entropy_weight_enable': self.distill_student_entropy_weight_enable,
                'distill_student_entropy_weight_mode': self.distill_student_entropy_weight_mode,
                'distill_student_entropy_weight_formula': self.distill_student_entropy_weight_formula,
                'distill_student_entropy_weight_min': self.distill_student_entropy_weight_min,
                'distill_student_entropy_weight_max': self.distill_student_entropy_weight_max,
                'distill_student_entropy_weight_beta': self.distill_student_entropy_weight_beta,
                'distill_student_entropy_weight_beta_pos': self.distill_student_entropy_weight_beta_pos,
                'distill_student_entropy_weight_beta_neg': self.distill_student_entropy_weight_beta_neg,
                'distill_student_entropy_weight_tau': self.distill_student_entropy_weight_tau,
                'distill_student_entropy_weight_normalize_mean': self.distill_student_entropy_weight_normalize_mean,
                'distill_student_entropy_weight_detach': self.distill_student_entropy_weight_detach,
                'distill_student_entropy_weight_target': self.distill_student_entropy_weight_target,
                'distill_student_entropy_weight_reg': self.distill_student_entropy_weight_reg,
                'distill_student_rgb_layers': self.distill_student_rgb_layers,
                'distill_student_ir_layers': self.distill_student_ir_layers,
                'distill_teacher_rgb_layers': self.distill_teacher_rgb_layers,
                'distill_teacher_ir_layers': self.distill_teacher_ir_layers,
                'distill_student_rgb_channels': self.distill_student_rgb_channels,
                'distill_student_ir_channels': self.distill_student_ir_channels,
                'distill_student_fusion_layers': self.distill_student_fusion_layers,
                'distill_student_fusion_channels': self.distill_student_fusion_channels,
                'distill_teacher_rgb_channels': self.distill_teacher_rgb_channels,
                'distill_teacher_ir_channels': self.distill_teacher_ir_channels,
                'distill_stage2_cls_only': self.distill_stage2_cls_only,
                'distill_resume_ckpt': self.distill_resume_ckpt,
                'freeze_backbone_fusion': self.freeze_backbone_fusion,
                'train_neck_head_only': self.train_neck_head_only,
                'stage2_lr_mult': self.stage2_lr_mult,
                'stage2_epochs': self.stage2_epochs,
                'stage2_disable_early_stop': self.stage2_disable_early_stop,
                'distill_cls_kd_weight': self.distill_cls_kd_weight,
                'distill_kd_temperature': self.distill_kd_temperature,
                'teacher_conf_thr': self.teacher_conf_thr,
                'teacher_entropy_thr': self.teacher_entropy_thr,
                'teacher_jsd_thr': self.teacher_jsd_thr,
            })
            if self.Distillation == "MultiDistillation":
                train_args['Teacher_Model_RGB_Path'] = str(self.teacher_model_rgb_path) if self.teacher_model_rgb_path is not None else None
                train_args['Teacher_Model_IR_Path'] = str(self.teacher_model_ir_path) if self.teacher_model_ir_path is not None else None
            else:
                train_args['Teacher_Model_Path'] = str(self.teacher_model_path) if self.teacher_model_path is not None else None
        model_to_save = self._clear_model_runtime_caches(self.model)
        ema_to_save = self._clear_model_runtime_caches(self.ema.ema)
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(model_to_save).half(),
            'ema': deepcopy(ema_to_save).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': train_args,  # save as dict
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Use dill (if exists) to serialize the lambda functions where pickle does not do this
        try:
            import dill as pickle
        except ImportError:
            import pickle

        # Save last and best
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        self._save_audit_checkpoint(ckpt)

    @staticmethod
    def get_dataset(data):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        return data['train'], data.get('val') or data.get('test')

    def _get_stage2_freeze_boundary(self):
        fusion_layers = self.distill_student_fusion_layers or []
        if fusion_layers:
            return max(int(idx) for idx in fusion_layers)
        branch_layers = (self.distill_student_rgb_layers or []) + (self.distill_student_ir_layers or [])
        if branch_layers:
            return max(int(idx) for idx in branch_layers)
        return -1

    def _apply_stage2_freeze(self):
        self.stage2_frozen_module_indices = []
        self.stage2_trainable_param_count = 0
        self.stage2_frozen_param_count = 0
        if not (self.distill_stage2_cls_only and self.freeze_backbone_fusion and self.train_neck_head_only):
            return

        top_modules = getattr(de_parallel(self.model), 'model', None)
        if top_modules is None:
            LOGGER.warning('Stage-2 freeze requested, but model has no top-level model list; skipping freeze.')
            return

        freeze_boundary = self._get_stage2_freeze_boundary()
        if freeze_boundary < 0:
            LOGGER.warning('Stage-2 freeze requested, but no backbone/fusion boundary was resolved; skipping freeze.')
            return

        self.stage2_frozen_module_indices = list(range(freeze_boundary + 1))
        for idx, module in enumerate(top_modules):
            should_train = idx > freeze_boundary
            if not should_train:
                module.eval()
            for param in module.parameters(recurse=True):
                if should_train:
                    # Preserve parameters intentionally frozen earlier in the setup flow
                    # (for example the fixed DFL projection conv).
                    if param.requires_grad:
                        self.stage2_trainable_param_count += param.numel()
                    else:
                        self.stage2_frozen_param_count += param.numel()
                    continue
                param.requires_grad = False
                if param.requires_grad:
                    self.stage2_trainable_param_count += param.numel()
                else:
                    self.stage2_frozen_param_count += param.numel()

        LOGGER.info(
            'Stage-2 cls-only mode: freezing backbone/fusion modules '
            f'0..{freeze_boundary} and training neck/head only '
            f'({self.stage2_trainable_param_count} trainable params, '
            f'{self.stage2_frozen_param_count} frozen params).')

    def _enforce_stage2_frozen_modules_eval(self):
        if not self.stage2_frozen_module_indices:
            return
        top_modules = getattr(de_parallel(self.model), 'model', None)
        if top_modules is None:
            return
        for idx in self.stage2_frozen_module_indices:
            if idx < len(top_modules):
                top_modules[idx].eval()

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if self.distill_stage2_cls_only and self.distill_resume_ckpt and not self.args.resume:
            model = str(self.distill_resume_ckpt)
            self.stage2_resume_ckpt_used = model
            LOGGER.info(f'Stage-2 cls-only mode loading recovery checkpoint from {model}')
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml

            # 是否蒸馏
            if self.Distillation is not None:
                self.model = ckpt['model']
                self.model.info()

        else:
            cfg = model

        pretrained = getattr(self.args, 'pretrained', None)
        if weights is None and isinstance(pretrained, str) and pretrained:
            LOGGER.info(f'Loading shape-matched pretrained weights from {pretrained}')
            weights, _ = attempt_load_one_weight(pretrained)

        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)

        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError('get_validator function not implemented in trainer')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError('get_dataloader function not implemented in trainer')

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build dataset."""
        raise NotImplementedError('build_dataset function not implemented in trainer')

    def label_loss_items(self, loss_items=None, prefix='train'):
        """Returns a loss dict with labelled training loss items tensor."""
        # Not needed for classification but necessary for segmentation & detection
        return {'loss': loss_items} if loss_items is not None else ['loss']

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data['names']

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ''

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch + 1] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {'data': data, 'timestamp': time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        # Keep last.pt resumable. Only strip best.pt for lightweight final inference/eval.
        if self.best.exists():
            strip_optimizer(self.best)
            LOGGER.info(f'\nValidating {self.best}...')
            self.validator.args.plots = self.args.plots
            self.metrics = self.validator(model=self.best)
            self.metrics.pop('fitness', None)
            self.run_callbacks('on_fit_epoch_end')

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args['data']).exists():
                    ckpt_args['data'] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = str(last)  # reinstate model
                preserved_override_keys = (
                    'imgsz', 'batch', 'epochs', 'patience', 'optimizer', 'lr0',
                    'device', 'workers', 'project', 'name', 'save_dir', 'exist_ok', 'split',
                    'Distillation', 'loss_type', 'distill_weight', 'online',
                    'Teacher_Model_RGB_Path', 'Teacher_Model_IR_Path', 'Teacher_Model_Path',
                    'distill_student_rgb_layers', 'distill_student_ir_layers',
                    'distill_teacher_rgb_layers', 'distill_teacher_ir_layers',
                    'distill_student_rgb_channels', 'distill_student_ir_channels',
                    'distill_student_fusion_layers', 'distill_student_fusion_channels',
                    'distill_teacher_rgb_channels', 'distill_teacher_ir_channels',
                    'distill_disable_all', 'distill_cross_attention', 'distill_normal_distillation',
                    'distill_only_normal',
                    'distill_disable_gate_kd', 'distill_disable_pseudo_fusion_kd',
                    'distill_head_kd_policy',
                    'distill_epoch_scale_mode', 'distill_epoch_scale_start', 'distill_epoch_scale_end',
                    'distill_epoch_scale_decay_start', 'distill_epoch_scale_decay_end',
                    'distill_zero_after_epoch',
                    'distill_student_entropy_weight_enable', 'distill_student_entropy_weight_mode',
                    'distill_student_entropy_weight_formula',
                    'distill_student_entropy_weight_min', 'distill_student_entropy_weight_max',
                    'distill_student_entropy_weight_beta',
                    'distill_student_entropy_weight_beta_pos', 'distill_student_entropy_weight_beta_neg',
                    'distill_student_entropy_weight_tau',
                    'distill_student_entropy_weight_normalize_mean',
                    'distill_student_entropy_weight_detach',
                    'distill_student_entropy_weight_target', 'distill_student_entropy_weight_reg',
                    'distill_schedule_enable', 'distill_mid_kd_start_epoch', 'distill_mid_kd_policy',
                    'distill_gate_kd_mode', 'distill_gate_kd_weight', 'distill_gate_kd_temperature',
                    'distill_gate_kd_mask_mode', 'distill_gate_kd_conf_thr',
                    'distill_late_kd_start_epoch', 'distill_late_kd_policy',
                    'distill_stage2_cls_only', 'distill_resume_ckpt',
                    'freeze_backbone_fusion', 'train_neck_head_only',
                    'stage2_lr_mult', 'stage2_epochs', 'stage2_disable_early_stop',
                    'distill_cls_kd_weight', 'distill_kd_temperature',
                    'teacher_conf_thr', 'teacher_entropy_thr', 'teacher_jsd_thr',
                    'audit_mode', 'audit_disable_early_stop', 'audit_ckpt_interval',
                    'audit_probe_epochs', 'audit_probe_fractions', 'audit_probes',
                    'audit_log_every_n', 'audit_output_name', 'audit_eval_split',
                )
                for k in preserved_override_keys:
                    if k in overrides:
                        setattr(self.args, k, overrides[k])
                if getattr(self, 'distill_disable_all', False):
                    self.args.Distillation = None
                    self.args.distill_weight = 0.0
                    self.args.online = False
                    self.args.distill_cross_attention = False
                    self.args.distill_normal_distillation = False
                    for path_key in ('Teacher_Model_RGB_Path', 'Teacher_Model_IR_Path', 'Teacher_Model_Path'):
                        if hasattr(self.args, path_key):
                            setattr(self.args, path_key, None)

            except Exception as e:
                raise FileNotFoundError('Resume checkpoint not found. Please pass a valid checkpoint to resume from, '
                                        "i.e. 'yolo train resume model=path/to/last.pt'") from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
                best_fitness = ckpt['best_fitness']
            except ValueError as exc:
                LOGGER.warning(
                    f'WARNING ⚠️ skipping optimizer state restore for {self.args.model} because parameter groups '
                    f'changed after resume overrides: {exc}')
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        self._set_train_augmentation(self.train_augment_target and start_epoch >= self.augment_start_epoch)
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _set_train_augmentation(self, enabled, reason=None):
        """Toggle train-time dataset augmentations and rebuild transforms when needed."""
        enabled = bool(enabled) and self.train_augment_target
        self.args.train_augment_active = enabled
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            return

        dataset = getattr(self.train_loader, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'build_transforms'):
            return
        if getattr(dataset, 'augment', None) == enabled:
            return

        dataset.augment = enabled
        dataset.max_buffer_length = min((dataset.ni, dataset.batch_size * 8, 1000)) if enabled else 0
        if hasattr(dataset, 'buffer') and not enabled:
            dataset.buffer.clear()
        dataset.transforms = dataset.build_transforms(hyp=self.args)
        if hasattr(self.train_loader, 'reset'):
            self.train_loader.reset()

        action = 'Enabling' if enabled else 'Disabling'
        if reason:
            LOGGER.info(f'{action} train augmentations ({reason})')
        else:
            LOGGER.info(f'{action} train augmentations')

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, 'mosaic'):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, 'close_mosaic'):
            LOGGER.info('Closing dataloader mosaic')
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer_multi(self, model, teacher_model_rgb, teacher_model_ir, distill_loss, name='auto', lr=0.001,
                              momentum=0.9, decay=1e-5, iterations=1e5):
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            LOGGER.info(f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                        f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                        f"determining best 'optimizer', 'lr0' and 'momentum' automatically... ")
            nc = getattr(model, 'nc', 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        ############################# 蒸馏 ############################
        if self.online:
            for v in teacher_model_rgb.modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                    if v.bias.requires_grad:
                        g[2].append(v.bias)
                if isinstance(v, bn):  # weight (no decay)
                    if v.weight.requires_grad:
                        g[1].append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                    if v.weight.requires_grad:
                        g[0].append(v.weight)

            for v in teacher_model_ir.modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                    if v.bias.requires_grad:
                        g[2].append(v.bias)
                if isinstance(v, bn):  # weight (no decay)
                    if v.weight.requires_grad:
                        g[1].append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                    if v.weight.requires_grad:
                        g[0].append(v.weight)

        if distill_loss is not None:
            for module_name, module in distill_loss.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    fullname = f'{module_name}.{param_name}' if module_name else param_name
                    if 'bias' in fullname:  # bias (no decay)
                        g[2].append(param)
                    elif isinstance(module, bn) or 'bn' in fullname:  # weight (no decay)
                        g[1].append(param)
                    else:  # weight (with decay)
                        g[0].append(param)
        ############################# 蒸馏 ############################

        unique_groups = []
        for params in g:
            unique_params = []
            seen = set()
            for param in params:
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                unique_params.append(param)
            unique_groups.append(unique_params)
        g = tuple(unique_groups)

        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)')
        return optimizer

    def build_optimizer(self, model, teacher_model, distill_loss, name='auto', lr=0.001, momentum=0.9, decay=1e-5,
                        iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            LOGGER.info(f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                        f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                        f"determining best 'optimizer', 'lr0' and 'momentum' automatically... ")
            nc = getattr(model, 'nc', 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        ############################# 蒸馏 ############################

        if self.Distillation is not None and self.online:
            for v in teacher_model.modules():
                # print(v)
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                    if v.bias.requires_grad:
                        g[2].append(v.bias)
                if isinstance(v, bn):  # weight (no decay)
                    if v.weight.requires_grad:
                        g[1].append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                    if v.weight.requires_grad:
                        g[0].append(v.weight)

        if self.Distillation is not None and distill_loss is not None:
            for k, v in distill_loss.named_parameters():
                if not v.requires_grad:
                    continue
                # print(v)
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                    g[2].append(v.bias)
                if isinstance(v, bn) or 'bn' in k:  # weight (no decay)
                    g[1].append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                    g[0].append(v.weight)

        ############################# 蒸馏 ############################

        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)')
        return optimizer
