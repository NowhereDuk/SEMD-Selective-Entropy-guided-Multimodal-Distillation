#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_YAML=${DATA_YAML:-$REPO_ROOT/data/DroneVehicle_obb_student_external.yaml}
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/runs/obb/semd_default/weights/best.pt}
PROJECT_DIR=${PROJECT_DIR:-$REPO_ROOT/runs/obb}
RUN_NAME=${RUN_NAME:-semd_default_test}
DEVICE=${DEVICE:-0,1,2,3}
WORKERS=${WORKERS:-8}
BATCH=${BATCH:-32}
IMGSZ=${IMGSZ:-640}
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export DATA_YAML MODEL_PATH PROJECT_DIR RUN_NAME DEVICE WORKERS BATCH IMGSZ

"$PYTHON_BIN" - <<'PY'
import os
from ultralytics import YOLO

model = YOLO(os.environ['MODEL_PATH'], task='obb')
model.val(
    data=os.environ['DATA_YAML'],
    split='test',
    imgsz=int(os.environ['IMGSZ']),
    batch=int(os.environ['BATCH']),
    device=os.environ['DEVICE'],
    workers=int(os.environ['WORKERS']),
    project=os.environ['PROJECT_DIR'],
    name=os.environ['RUN_NAME'],
    exist_ok=True,
)
PY
