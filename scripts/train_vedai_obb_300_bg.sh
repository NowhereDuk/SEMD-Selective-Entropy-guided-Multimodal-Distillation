#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-semd}

DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/datasets/VEDAI_obb}
DATA_TEACHER_RGB=${DATA_TEACHER_RGB:-$DATA_ROOT/VEDAI_teacher_rgb_obb.yaml}
DATA_TEACHER_IR=${DATA_TEACHER_IR:-$DATA_ROOT/VEDAI_teacher_ir_obb.yaml}
DATA_STUDENT=${DATA_STUDENT:-$REPO_ROOT/data/VEDAI_obb_student_external.yaml}

MODEL_TEACHER=${MODEL_TEACHER:-$REPO_ROOT/model_yaml_obb/yolov8-hgnetv2-b0-teacher-obb.yaml}
MODEL_STUDENT=${MODEL_STUDENT:-$REPO_ROOT/model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml}

DEVICE=${DEVICE:-4,5,6,7}
WORKERS=${WORKERS:-24}
IMGSZ=${IMGSZ:-1024}

EPOCHS_TEACHER=${EPOCHS_TEACHER:-300}
EPOCHS_STUDENT=${EPOCHS_STUDENT:-300}
PATIENCE_TEACHER=${PATIENCE_TEACHER:-120}
PATIENCE_STUDENT=${PATIENCE_STUDENT:-120}
BATCH_TEACHER=${BATCH_TEACHER:-80}
BATCH_STUDENT=${BATCH_STUDENT:-24}
RUN_TAG=${RUN_TAG:-img1024_bt80_bs24}

TEACHER_RGB_DIR=${TEACHER_RGB_DIR:-$REPO_ROOT/runs/obb/teacher_hgnetv2_vedai_obb_rgb_ep300_${RUN_TAG}}
TEACHER_IR_DIR=${TEACHER_IR_DIR:-$REPO_ROOT/runs/obb/teacher_hgnetv2_vedai_obb_ir_ep300_${RUN_TAG}}
STUDENT_DIR=${STUDENT_DIR:-$REPO_ROOT/runs/obb/student_vedai_obb_ep300_${RUN_TAG}}

LOG_DIR=${LOG_DIR:-$REPO_ROOT/runs/logs}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-$LOG_DIR/vedai_obb_pipeline_ep300_${RUN_TAG}_${STAMP}.log}

run_pipeline() {
  cd "$REPO_ROOT"
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

  echo "[info] start time: $(date -Iseconds)"
  echo "[info] repo: $REPO_ROOT"
  echo "[info] data root: $DATA_ROOT"
  echo "[info] device: $DEVICE"
  echo "[info] imgsz: $IMGSZ"
  echo "[info] teacher batch: $BATCH_TEACHER"
  echo "[info] student batch: $BATCH_STUDENT"
  echo "[info] run tag: $RUN_TAG"
  echo "[info] pythonpath: $PYTHONPATH"
  echo "[info] teacher epochs: $EPOCHS_TEACHER"
  echo "[info] student epochs: $EPOCHS_STUDENT"

  python "$REPO_ROOT/train_teacher_hgnetv2_obb.py" \
    --modality rgb \
    --data "$DATA_TEACHER_RGB" \
    --model "$MODEL_TEACHER" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --batch "$BATCH_TEACHER" \
    --imgsz "$IMGSZ" \
    --epochs "$EPOCHS_TEACHER" \
    --patience "$PATIENCE_TEACHER" \
    --save-dir "$TEACHER_RGB_DIR" \
    --optimizer auto \
    --lr0 0.001 \
    --augment \
    --exist-ok

  test -f "$TEACHER_RGB_DIR/weights/best.pt"

  python "$REPO_ROOT/train_teacher_hgnetv2_obb.py" \
    --modality ir \
    --data "$DATA_TEACHER_IR" \
    --model "$MODEL_TEACHER" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --batch "$BATCH_TEACHER" \
    --imgsz "$IMGSZ" \
    --epochs "$EPOCHS_TEACHER" \
    --patience "$PATIENCE_TEACHER" \
    --save-dir "$TEACHER_IR_DIR" \
    --optimizer auto \
    --lr0 0.001 \
    --augment \
    --exist-ok

  test -f "$TEACHER_IR_DIR/weights/best.pt"

  python "$REPO_ROOT/train_student_deimhgnetv2_obb.py" \
    --data "$DATA_STUDENT" \
    --model "$MODEL_STUDENT" \
    --save-dir "$STUDENT_DIR" \
    --teacher-arch hgnetv2 \
    --teacher-rgb "$TEACHER_RGB_DIR/weights/best.pt" \
    --teacher-ir "$TEACHER_IR_DIR/weights/best.pt" \
    --epochs "$EPOCHS_STUDENT" \
    --patience "$PATIENCE_STUDENT" \
    --batch "$BATCH_STUDENT" \
    --imgsz "$IMGSZ" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --optimizer AdamW \
    --lr0 0.0005 \
    --augment \
    --no-amp \
    --distill-weight 0.6 \
    --distill-cross-attention \
    --distill-normal-distillation \
    --distill-head-kd-policy off \
    --distill-disable-pseudo-fusion-kd \
    --distill-gate-kd-mode legacy \
    --distill-gate-kd-weight 0.7 \
    --distill-gate-kd-temperature 1.0 \
    --distill-gate-kd-mask-mode none \
    --distill-gate-kd-conf-thr 0.25 \
    --distill-cls-kd-weight 0.03 \
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

  echo "[info] finished time: $(date -Iseconds)"
}

launch_background() {
  nohup bash "$0" --worker >"$LOG_FILE" 2>&1 &
  echo "PID=$!"
  echo "LOG=$LOG_FILE"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_pipeline
else
  launch_background
fi
