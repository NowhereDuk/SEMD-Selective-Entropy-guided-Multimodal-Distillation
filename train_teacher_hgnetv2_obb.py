#!/usr/bin/env python3
"""Train a single-modal HGNetv2-B0 OBB teacher from the main repository ultralytics."""

import argparse
import os
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ultralytics.models.yolo.obb import OBBTrainer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "model_yaml_obb" / "yolov8-hgnetv2-b0-obb-teacher.yaml"
DEFAULT_DATA_YAMLS = {
    "rgb": REPO_ROOT / "teacherTraining" / "Drone_RGB_obb_external.yaml",
    "ir": REPO_ROOT / "teacherTraining" / "Drone_IR_obb_external.yaml",
}
DEFAULT_PROJECT = REPO_ROOT / "runs" / "obb"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-modal HGNetv2-B0 OBB teacher.")
    parser.add_argument("--modality", choices=("rgb", "ir"), default="rgb")
    parser.add_argument("--data", type=str, default=None, help="Optional data YAML override.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--project", type=str, default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_data_yaml(modality, data_override=None):
    return resolve_path(data_override) if data_override else DEFAULT_DATA_YAMLS[modality]


def resolve_dataset_root(data_yaml_path):
    data_yaml_path = resolve_path(data_yaml_path)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data yaml not found: {data_yaml_path}")
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    raw_root = str(data.get("path", "")).strip()
    if not raw_root:
        raise ValueError(f"dataset yaml missing 'path': {data_yaml_path}")
    dataset_root = Path(raw_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()
    return data_yaml_path, dataset_root


def first_non_empty_label(label_dir):
    for label_path in sorted(label_dir.glob("*.txt")):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return label_path, line.strip()
    return None, None


def validate_obb_dataset(data_yaml_path):
    data_yaml_path, dataset_root = resolve_dataset_root(data_yaml_path)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    checked_split = False
    for split in ("train", "val"):
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            continue
        checked_split = True
        label_path, sample_line = first_non_empty_label(label_dir)
        if sample_line is None:
            continue
        if len(sample_line.split()) < 9:
            raise ValueError(
                f"dataset is not OBB-ready: {label_path} has fewer than 9 columns "
                f"(expected class + 8 coordinates)."
            )
    if not checked_split:
        raise FileNotFoundError(f"no labels/train or labels/val directories found under {dataset_root}")
    return data_yaml_path, dataset_root


def main():
    args = parse_args()
    model_path = resolve_path(args.model)
    data_path, dataset_root = validate_obb_dataset(resolve_data_yaml(args.modality, args.data))
    run_name = args.name or f"teacher_hgnetv2_obb_{args.modality}"

    if not model_path.exists():
        raise FileNotFoundError(f"model yaml not found: {model_path}")

    print(f"[info] modality: {args.modality}")
    print(f"[info] model: {model_path}")
    print(f"[info] data: {data_path}")
    print(f"[info] dataset root: {dataset_root}")
    print(f"[info] run name: {run_name}")

    overrides = dict(
        model=str(model_path),
        data=str(data_path),
        task="obb",
        amp=args.amp,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        optimizer=args.optimizer,
        augment=args.augment,
        workers=args.workers,
        rect=args.rect,
        project=str(resolve_path(args.project)),
        name=run_name,
        exist_ok=args.exist_ok,
        ch=3,
    )

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
