# SEMD Method Summary

SEMD, Selective Entropy-guided Multimodal Distillation, is designed for RGB-IR
oriented object detection. It trains a compact dual-stream student detector with
two single-modality teachers and a spatially adaptive fusion gate.

## Core Components

- **Dual-stream student:** RGB and infrared images are processed by separate
  branches and fused at multi-scale feature levels.
- **Entropy-guided fusion:** `EntropyPrior` builds a high-resolution prior from
  the paired input, while `EntropyOffsetGateFusion` aligns features and predicts
  RGB/IR gate probabilities for each fusion stage.
- **Dual-teacher distillation:** the RGB teacher and IR teacher supervise the
  corresponding student branches and provide cross-modal reliability cues.
- **Gate KD:** teacher confidence maps are converted into a two-channel modality
  preference target. The student fusion gate is trained to match this target.
- **Student entropy weighting:** the student gate entropy measures uncertainty
  in modality selection. Uncertain regions receive stronger gate supervision:

```text
w = clip(1 + beta_pos * relu(H - tau) - beta_neg * relu(tau - H), w_min, w_max)
```

The default paper setting uses `tau=0.5`, `beta_pos=1.0`, `beta_neg=0.5`,
`w_min=0.5`, and `w_max=1.5`.

## Recommended Training Switches

```bash
python train_student_deimhgnetv2_obb.py \
  --data data/DroneVehicle_obb_student_external.yaml \
  --model model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml \
  --teacher-rgb weight/teacher_hgnetv2_obb_rgb/best.pt \
  --teacher-ir weight/teacher_hgnetv2_obb_ir/best.pt \
  --distill-cross-attention \
  --distill-normal-distillation \
  --distill-head-kd-policy off \
  --distill-disable-pseudo-fusion-kd \
  --distill-gate-kd-mode legacy \
  --distill-gate-kd-weight 1.0 \
  --distill-student-entropy-weight-enable \
  --distill-student-entropy-weight-mode fixed \
  --distill-student-entropy-weight-formula asym_centered \
  --distill-student-entropy-weight-target gate
```

This setting focuses the distillation objective on branch features,
cross-modal features, and gate supervision. Detection-head KD and pseudo-fusion
KD are disabled in the default release recipe.

## Qualitative Visualization

Use `scripts/generate_semd_qualitative.py` after preparing SEMD and teacher
checkpoints. The script can produce detection comparisons, student/teacher
modality preference maps, and student gate entropy maps for paper figures.
