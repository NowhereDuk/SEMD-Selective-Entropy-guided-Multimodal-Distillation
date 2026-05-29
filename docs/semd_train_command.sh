#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_YAML=${DATA_YAML:-$REPO_ROOT/data/DroneVehicle_obb_student_external.yaml}
MODEL_YAML=${MODEL_YAML:-$REPO_ROOT/model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml}
TEACHER_RGB=${TEACHER_RGB:-$REPO_ROOT/weight/teacher_hgnetv2_obb_rgb/best.pt}
TEACHER_IR=${TEACHER_IR:-$REPO_ROOT/weight/teacher_hgnetv2_obb_ir/best.pt}
SAVE_DIR=${SAVE_DIR:-$REPO_ROOT/runs/obb/semd_default}
DEVICE=${DEVICE:-0,1,2,3}
WORKERS=${WORKERS:-8}
BATCH=${BATCH:-32}
IMGSZ=${IMGSZ:-640}
EPOCHS=${EPOCHS:-132}
PATIENCE=${PATIENCE:-50}
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" "$REPO_ROOT/train_student_deimhgnetv2_obb.py" \
  --data "$DATA_YAML" \
  --model "$MODEL_YAML" \
  --save-dir "$SAVE_DIR" \
  --teacher-arch hgnetv2 \
  --teacher-rgb "$TEACHER_RGB" \
  --teacher-ir "$TEACHER_IR" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
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
