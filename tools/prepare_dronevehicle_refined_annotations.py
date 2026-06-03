#!/usr/bin/env python3
"""Install SEMD refined DroneVehicle labels into a local dataset root."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_ROOT = REPO_ROOT / "annotations" / "dronevehicle_refined_yolo_obb" / "labels"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local DroneVehicle-style dataset root containing images/ and images_ir/.",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
        help="Refined YOLO-OBB label root to install.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Copy labels or create split-level symlinks under dataset-root/labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing dataset-root/labels/<split> directories or symlinks.",
    )
    parser.add_argument(
        "--write-yaml",
        type=Path,
        default=None,
        help="Optional path for a generated Ultralytics dataset YAML.",
    )
    return parser.parse_args()


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def install_split(src: Path, dst: Path, mode: str, overwrite: bool) -> int:
    require_dir(src)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        remove_existing(dst)

    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
    else:
        shutil.copytree(src, dst)
    return sum(1 for _ in dst.glob("*.txt"))


def write_yaml(path: Path, dataset_root: Path) -> None:
    text = f"""# SEMD refined DroneVehicle annotations in YOLO OBB format.
path: {dataset_root.resolve()}

train: images/train
val: images/val
test: images/test

nc: 5
ch: 6

names:
  0: car
  1: freight_car
  2: truck
  3: bus
  4: van
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    label_root = args.label_root

    require_dir(label_root)
    require_dir(dataset_root / "images")
    require_dir(dataset_root / "images_ir")

    target_label_root = dataset_root / "labels"
    target_label_root.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in SPLITS:
        counts[split] = install_split(
            label_root / split,
            target_label_root / split,
            args.mode,
            args.overwrite,
        )

    if args.write_yaml is not None:
        write_yaml(args.write_yaml, dataset_root)

    print("Installed SEMD refined DroneVehicle labels:")
    for split in SPLITS:
        print(f"  {split}: {counts[split]} label files")
    print(f"  target: {target_label_root}")
    if args.write_yaml is not None:
        print(f"  yaml: {args.write_yaml}")


if __name__ == "__main__":
    main()
