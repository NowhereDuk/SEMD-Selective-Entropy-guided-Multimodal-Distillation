#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-semd}

CV_ROOT=${CV_ROOT:-$REPO_ROOT/datasets/VEDAI_obb_cv}
FOLDS=${FOLDS:-01,02,03,04,05,06,07,08,09,10}
DEVICE=${DEVICE:-4,5,6,7}
WORKERS=${WORKERS:-24}
IMGSZ=${IMGSZ:-1024}

MODEL_TEACHER=${MODEL_TEACHER:-$REPO_ROOT/model_yaml_obb/yolov8-hgnetv2-b0-teacher-obb.yaml}
MODEL_STUDENT=${MODEL_STUDENT:-$REPO_ROOT/model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml}

EPOCHS_TEACHER=${EPOCHS_TEACHER:-300}
EPOCHS_STUDENT=${EPOCHS_STUDENT:-300}
PATIENCE_TEACHER=${PATIENCE_TEACHER:-120}
PATIENCE_STUDENT=${PATIENCE_STUDENT:-120}
BATCH_TEACHER=${BATCH_TEACHER:-80}
BATCH_STUDENT=${BATCH_STUDENT:-24}
RUN_TAG=${RUN_TAG:-img1024_bt80_bs24}

LOG_DIR=${LOG_DIR:-$REPO_ROOT/runs/logs}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG=${MASTER_LOG:-$LOG_DIR/vedai_obb_cv10_ep300_${RUN_TAG}_${STAMP}.log}

has_valid_best() {
  local run_dir=$1
  [[ -s "$run_dir/weights/best.pt" ]]
}

run_batch() {
  cd "$REPO_ROOT"
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

  IFS=',' read -r -a fold_list <<<"$FOLDS"

  echo "[info] start time: $(date -Iseconds)"
  echo "[info] repo: $REPO_ROOT"
  echo "[info] cv root: $CV_ROOT"
  echo "[info] folds: ${fold_list[*]}"
  echo "[info] device: $DEVICE"
  echo "[info] imgsz: $IMGSZ"
  echo "[info] teacher batch: $BATCH_TEACHER"
  echo "[info] student batch: $BATCH_STUDENT"
  echo "[info] run tag: $RUN_TAG"
  echo "[info] pythonpath: $PYTHONPATH"

  for fold_id in "${fold_list[@]}"; do
    fold_id=$(printf "%02d" "$((10#$fold_id))")
    fold_root="$CV_ROOT/fold${fold_id}"
    fold_log="$LOG_DIR/vedai_obb_fold${fold_id}_ep300_${RUN_TAG}_${STAMP}.log"
    teacher_rgb_dir="$REPO_ROOT/runs/obb_cv/fold${fold_id}/teacher_rgb_ep300_${RUN_TAG}"
    teacher_ir_dir="$REPO_ROOT/runs/obb_cv/fold${fold_id}/teacher_ir_ep300_${RUN_TAG}"
    student_dir="$REPO_ROOT/runs/obb_cv/fold${fold_id}/student_ep300_${RUN_TAG}"

    echo ""
    echo "[info] ===== fold${fold_id} start: $(date -Iseconds) ====="
    echo "[info] fold${fold_id} log: $fold_log"

    {
      echo "[info] fold${fold_id} start time: $(date -Iseconds)"
      echo "[info] fold root: $fold_root"
      echo "[info] device: $DEVICE"
      echo "[info] imgsz: $IMGSZ"
      echo "[info] teacher batch: $BATCH_TEACHER"
      echo "[info] student batch: $BATCH_STUDENT"
      echo "[info] run tag: $RUN_TAG"

      if has_valid_best "$teacher_rgb_dir"; then
        echo "[info] skip teacher rgb: existing best.pt at $teacher_rgb_dir/weights/best.pt"
      else
        python "$REPO_ROOT/train_teacher_hgnetv2_obb.py" \
          --modality rgb \
          --data "$fold_root/VEDAI_teacher_rgb_obb.yaml" \
          --model "$MODEL_TEACHER" \
          --device "$DEVICE" \
          --workers "$WORKERS" \
          --batch "$BATCH_TEACHER" \
          --imgsz "$IMGSZ" \
          --epochs "$EPOCHS_TEACHER" \
          --patience "$PATIENCE_TEACHER" \
          --save-dir "$teacher_rgb_dir" \
          --optimizer auto \
          --lr0 0.001 \
          --augment \
          --exist-ok
      fi

      test -s "$teacher_rgb_dir/weights/best.pt"

      if has_valid_best "$teacher_ir_dir"; then
        echo "[info] skip teacher ir: existing best.pt at $teacher_ir_dir/weights/best.pt"
      else
        python "$REPO_ROOT/train_teacher_hgnetv2_obb.py" \
          --modality ir \
          --data "$fold_root/VEDAI_teacher_ir_obb.yaml" \
          --model "$MODEL_TEACHER" \
          --device "$DEVICE" \
          --workers "$WORKERS" \
          --batch "$BATCH_TEACHER" \
          --imgsz "$IMGSZ" \
          --epochs "$EPOCHS_TEACHER" \
          --patience "$PATIENCE_TEACHER" \
          --save-dir "$teacher_ir_dir" \
          --optimizer auto \
          --lr0 0.001 \
          --augment \
          --exist-ok
      fi

      test -s "$teacher_ir_dir/weights/best.pt"

      if has_valid_best "$student_dir"; then
        echo "[info] skip student: existing best.pt at $student_dir/weights/best.pt"
      else
        python "$REPO_ROOT/train_student_deimhgnetv2_obb.py" \
          --data "$fold_root/VEDAI_student_obb.yaml" \
          --model "$MODEL_STUDENT" \
          --save-dir "$student_dir" \
          --teacher-arch hgnetv2 \
          --teacher-rgb "$teacher_rgb_dir/weights/best.pt" \
          --teacher-ir "$teacher_ir_dir/weights/best.pt" \
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
      fi

      echo "[info] fold${fold_id} finished time: $(date -Iseconds)"
    } >"$fold_log" 2>&1

    echo "[info] ===== fold${fold_id} done: $(date -Iseconds) ====="
  done

  echo ""
  echo "[info] all folds finished: $(date -Iseconds)"
}

launch_background() {
  nohup bash "$0" --worker >"$MASTER_LOG" 2>&1 &
  echo "PID=$!"
  echo "LOG=$MASTER_LOG"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_batch
else
  launch_background
fi
