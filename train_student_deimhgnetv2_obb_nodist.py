#!/usr/bin/env python3
"""Train the dual-stream HGNetv2 OBB student without distillation."""

import argparse
import os
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils import DEFAULT_CFG


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "model_yaml_obb" / "yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "DroneVehicle_obb_student_external.yaml"
DEFAULT_SAVE_DIR = REPO_ROOT / "runs" / "obb" / "dronevehicle_student_nodist_entropygate_deimhgnetv2_obb"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the dual-stream HGNetv2 OBB student without distillation."
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a saved checkpoint path such as runs/.../weights/last.pt.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=132)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--optimizer",
        choices=("SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSProp", "auto"),
        default="auto",
    )
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_yaml_dict(path_str):
    yaml_path = resolve_path(path_str)
    if not yaml_path.exists():
        raise FileNotFoundError(f"yaml not found: {yaml_path}")
    return yaml_path, yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}


def resolve_dataset_root(data_yaml_path):
    data_yaml_path, data = load_yaml_dict(data_yaml_path)
    raw_root = str(data.get("path", "")).strip()
    if not raw_root:
        raise ValueError(f"dataset yaml missing 'path': {data_yaml_path}")
    dataset_root = Path(raw_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()
    return data_yaml_path, data, dataset_root


def first_non_empty_label(label_dir):
    for label_path in sorted(label_dir.glob("*.txt")):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return label_path, line.strip()
    return None, None


def validate_obb_dataset(data_yaml_path):
    data_yaml_path, data, dataset_root = resolve_dataset_root(data_yaml_path)
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
    return data_yaml_path, data, dataset_root


def validate_model_yaml(model_path):
    model_path = resolve_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"student model yaml not found: {model_path}")
    return model_path


def main():
    args = parse_args()
    data_yaml_path, data_cfg, dataset_root = validate_obb_dataset(args.data)
    model_path = validate_model_yaml(args.model)
    save_dir = resolve_path(args.save_dir)

    print(f"[info] student model: {model_path}")
    print(f"[info] data: {data_yaml_path}")
    print(f"[info] dataset root: {dataset_root}")
    print(f"[info] nc: {data_cfg.get('nc')}")
    print(f"[info] save dir: {save_dir}")

    overrides = dict(
        model=str(model_path),
        data=str(data_yaml_path),
        task="obb",
        resume=str(resolve_path(args.resume)) if args.resume else False,
        amp=args.amp,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        augment=args.augment,
        workers=args.workers,
        rect=args.rect,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=args.exist_ok,
        ch=6,
    )

    DEFAULT_CFG.save_dir = str(save_dir)
    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
