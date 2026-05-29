#!/usr/bin/env python3
"""Train a single-stream HGNetv2 HBB teacher for RGB-only or IR-only detect datasets."""

import argparse
import os
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-hgnetv2-b0-teacher.yaml"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "LLVIP_hbb_student_external.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-stream HGNetv2 HBB teacher detector.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a checkpoint path such as runs/.../weights/last.pt.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Detect/HBB dataset yaml for a single-modality teacher.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Teacher model yaml. Defaults to the existing HGNetv2 detect teacher yaml.")
    parser.add_argument("--modality", choices=("rgb", "ir"), required=True,
                        help="Teacher modality. Used for save-dir naming and dataset sanity checks.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Optional save dir. Defaults to runs/detect/teacher_hgnetv2_hbb_{modality}.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=132)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--optimizer",
        choices=("SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSProp", "auto"),
        default="auto",
    )
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--rect", action="store_true", default=False)
    parser.add_argument("--exist-ok", action="store_true", default=True)
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_yaml_dict(path_str):
    yaml_path = resolve_path(path_str)
    if not yaml_path.exists():
        raise FileNotFoundError(f"yaml not found: {yaml_path}")
    return yaml_path, yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}


def validate_teacher_dataset(data_yaml_path):
    data_yaml_path, data = load_yaml_dict(data_yaml_path)
    raw_root = str(data.get("path", "")).strip()
    if not raw_root:
        raise ValueError(f"dataset yaml missing 'path': {data_yaml_path}")

    ch = int(data.get("ch", 3))
    if ch != 3:
        raise ValueError(
            f"teacher dataset yaml must be single-modality with ch: 3, got ch: {ch} in {data_yaml_path}"
        )

    train_value = str(data.get("train", ""))
    val_value = str(data.get("val", ""))
    if not train_value or not val_value:
        raise ValueError(f"dataset yaml must define train and val: {data_yaml_path}")
    return data_yaml_path, data


def validate_teacher_model(model_path):
    model_path, model_cfg = load_yaml_dict(model_path)
    if int(model_cfg.get("ch", 3)) != 3:
        raise ValueError(f"teacher model yaml must be 3-channel, got ch={model_cfg.get('ch')} in {model_path}")
    return model_path, model_cfg


def main():
    args = parse_args()
    data_yaml_path, data_cfg = validate_teacher_dataset(args.data)
    model_path, model_cfg = validate_teacher_model(args.model)

    default_save_dir = REPO_ROOT / "runs" / "detect" / f"teacher_hgnetv2_hbb_{args.modality}"
    save_dir = resolve_path(args.save_dir) if args.save_dir else default_save_dir
    effective_resume = str(resolve_path(args.resume)) if args.resume else False

    print(f"[info] teacher modality: {args.modality}")
    print(f"[info] model: {model_path}")
    print(f"[info] data: {data_yaml_path}")
    print(f"[info] nc: {data_cfg.get('nc')} names: {data_cfg.get('names')}")
    print(f"[info] model ch: {model_cfg.get('ch')} data ch: {data_cfg.get('ch')}")
    print(f"[info] save dir: {save_dir}")

    overrides = dict(
        model=str(model_path),
        data=str(data_yaml_path),
        task="detect",
        resume=effective_resume,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        amp=args.amp,
        rect=args.rect,
        augment=args.augment,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=args.exist_ok,
        ch=3,
    )
    if args.patience is not None:
        overrides["patience"] = args.patience

    from ultralytics.models.yolo.detect import DetectionTrainer

    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
