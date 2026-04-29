# SEMD：面向 RGB-红外旋转车辆检测的选择性熵引导多模态蒸馏

本仓库实现了 **SEMD（Selective Entropy-guided Multimodal Distillation）**，一种面向 RGB-红外多模态旋转目标检测（Oriented Object Detection, OBB）的选择性熵引导多模态蒸馏方法。该方法用于训练双流 RGB-IR OBB 学生检测器，通过两个单模态教师模型分别提供 RGB 与红外监督，并重点优化学生网络在多模态融合阶段的门控决策。

与将所有教师信息无差别蒸馏到学生网络不同，SEMD 采用更精简、更稳定的蒸馏结构，仅保留：

- **分支特征蒸馏**：RGB 学生分支对齐 RGB 教师，IR 学生分支对齐 IR 教师；
- **跨模态蒸馏**：RGB 分支学习 IR 教师的互补信息，IR 分支学习 RGB 教师的互补信息；
- **Gate KD**：利用 RGB / IR 教师置信度构造融合门控软目标，指导学生在不同空间位置选择更可信的模态；
- **学生熵感知蒸馏强度控制**：根据学生 gate 输出的归一化熵，自适应增强或减弱 Gate KD 的空间蒸馏强度。

在论文对齐版本 `Exp2h` 中，方法显式关闭了 pseudo fusion feature KD 以及检测头蒸馏项（包括 `cls`、`DFL` 和 `angle`），从而将蒸馏重点集中在分支级、跨模态级和融合门控级监督上。该设计表明：对于 RGB-IR OBB 检测，蒸馏并非越多越好，选择合适的蒸馏分支和空间位置更加关键。

## 核心思想

SEMD 的核心是：在多模态蒸馏中，不同分支、不同空间位置和不同学生状态并不需要相同强度的教师约束。学生融合门控的熵可以反映模型对 RGB / IR 模态选择的不确定性：

```text
学生 gate entropy 高  -> 学生不确定该相信 RGB 还是 IR，应加强教师指导
学生 gate entropy 低  -> 学生已有明确融合决策，应适当减弱教师约束
```

因此，本文采用非对称 centered 熵权重：

```text
pos   = relu(H_gate - tau)
neg   = relu(tau - H_gate)
w_ent = clamp(1 + beta_pos * pos - beta_neg * neg, min_w, max_w)
```

其中 `H_gate` 是学生 gate 的归一化熵。最终 Gate KD 权重由教师置信度和学生熵权重共同决定：

```text
w_gate = w_conf * w_ent
```

这种设计使教师模型在学生不确定的位置提供更强指导，同时避免在学生已经形成稳定融合判断的位置产生过度约束。

## 方法概述

论文对齐版本采用如下蒸馏目标：

```text
L_distill = L_branch + L_cross + L_gate
```

其中：

```text
L_branch = L_rgb->rgb + L_ir->ir
L_cross  = L_ir->rgb + L_rgb->ir
```

总训练目标为：

```text
L_total = L_det + lambda(e) * (L_branch + L_cross + L_gate)
```

其中 `L_det` 表示原始 OBB 检测监督损失，`lambda(e)` 表示训练过程中的全局蒸馏权重。需要注意的是，学生熵控制只作用于 `Gate KD`，不会作用于原始检测监督损失，也不会作用于 `cls`、`DFL` 或 `angle` 等检测头蒸馏项。

## 实验结果

在 DroneVehicle RGB-IR OBB 测试集上，当前论文对齐版本取得如下结果：

| 指标 | 结果 |
|---|---:|
| Precision | 0.824 |
| Recall | 0.822 |
| mAP50 | 0.853 |
| mAP50-95 | 0.713 |

逐类别结果如下：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| car | 0.935 | 0.971 | 0.980 | 0.829 |
| freight_car | 0.645 | 0.754 | 0.713 | 0.581 |
| truck | 0.822 | 0.787 | 0.846 | 0.689 |
| bus | 0.917 | 0.933 | 0.956 | 0.815 |
| van | 0.801 | 0.666 | 0.767 | 0.648 |

## 训练入口

论文对齐版本的主要训练脚本为：

```bash
train_student_deimhgnetv2_obb.py
```

核心训练设置包括：

```bash
--distill-cross-attention
--distill-normal-distillation
--distill-head-kd-policy off
--distill-disable-pseudo-fusion-kd
--distill-gate-kd-mode legacy
--distill-student-entropy-weight-enable
--distill-student-entropy-weight-mode fixed
--distill-student-entropy-weight-formula asym_centered
--distill-student-entropy-weight-target gate
--distill-student-entropy-weight-detach
```

## 复现实验

示例训练命令如下，请根据本地路径修改 `data`、`model`、`teacher-rgb`、`teacher-ir` 和 `save-dir`：

```bash
python train_student_deimhgnetv2_obb.py \
  --data data/DroneVehicle_obb_student_external.yaml \
  --model model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml \
  --save-dir runs/obb/semd_exp2h \
  --teacher-arch hgnetv2 \
  --teacher-rgb weight/teacher_hgnetv2_obb_rgb/best.pt \
  --teacher-ir weight/teacher_hgnetv2_obb_ir/best.pt \
  --epochs 132 \
  --patience 50 \
  --batch 32 \
  --imgsz 640 \
  --device 0,1,2,3 \
  --workers 8 \
  --optimizer auto \
  --lr0 0.001 \
  --augment \
  --no-amp \
  --distill-cross-attention \
  --distill-normal-distillation \
  --distill-head-kd-policy off \
  --distill-disable-pseudo-fusion-kd \
  --distill-gate-kd-mode legacy \
  --distill-gate-kd-weight 1.0 \
  --distill-gate-kd-temperature 1.0 \
  --distill-gate-kd-mask-mode none \
  --distill-gate-kd-conf-thr 0.25 \
  --distill-cls-kd-weight 0.05 \
  --distill-kd-temperature 2.0 \
  --teacher-conf-thr 0.45 \
  --teacher-entropy-thr 0.35 \
  --teacher-jsd-thr 0.1 \
  --distill-student-entropy-weight-enable \
  --distill-student-entropy-weight-mode fixed \
  --distill-student-entropy-weight-formula asym_centered \
  --distill-student-entropy-weight-tau 0.5 \
  --distill-student-entropy-weight-beta-pos 1.0 \
  --distill-student-entropy-weight-beta-neg 0.5 \
  --distill-student-entropy-weight-min 0.5 \
  --distill-student-entropy-weight-max 1.5 \
  --distill-student-entropy-weight-target gate \
  --distill-student-entropy-weight-detach \
  --exist-ok
```

## 说明

本项目仍在整理中，后续将继续补充：

- 训练与测试完整命令；
- 数据集准备说明；
- 模型配置文件；
- 预训练权重；
- 消融实验表格；
- 可视化结果；
- 论文引用信息。

## 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{semd2026,
  title  = {Selective Entropy-guided Multimodal Distillation for RGB-Infrared Oriented Vehicle Detection},
  author = {Fan, Cunzheng},
  year   = {2026},
  note   = {GitHub repository}
}
```
