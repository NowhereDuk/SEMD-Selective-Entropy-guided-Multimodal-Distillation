#!/usr/bin/env python3
"""Convert raw VEDAI rotated annotations into 10-fold paired YOLO OBB layouts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dataset_conversion_utils import ConversionStats, ensure_output_dirs, materialize_file, write_multimodal_yaml, write_summary


IMG_W = 1024.0
IMG_H = 1024.0
CLASS_NAMES = ["plane", "boat", "camping_car", "car", "pickup", "tractor", "truck", "van", "other"]
RAW2NAME = {
    1: "car",
    2: "truck",
    4: "tractor",
    5: "camping_car",
    7: "other",
    8: "other",
    9: "van",
    10: "other",
    11: "pickup",
    12: "other",
    23: "boat",
    31: "plane",
}
NAME2ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw VEDAI to official 10-fold paired YOLO OBB layouts.")
    parser.add_argument("--src", default="datasets/raw/VEDAI", help="Raw VEDAI root.")
    parser.add_argument("--dst", default="datasets/VEDAI_obb_cv", help="Output dataset root for all folds.")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize paired images in the output dataset.",
    )
    parser.add_argument(
        "--folds",
        default="all",
        help="Comma-separated fold ids such as '01,02,10', or 'all' for fold01..fold10.",
    )
    parser.add_argument(
        "--val-mode",
        choices=("test_as_val",),
        default="test_as_val",
        help="How to define validation split for each official fold.",
    )
    parser.add_argument("--force", action="store_true", help="Replace destination if it already exists.")
    return parser.parse_args()


def parse_fold_ids(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return [f"{i:02d}" for i in range(1, 11)]
    fold_ids = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        fold_ids.append(f"{int(item):02d}")
    if not fold_ids:
        raise ValueError("no valid folds requested")
    return sorted(set(fold_ids))


def normalize_point(x: float, y: float, stats: ConversionStats) -> tuple[float, float]:
    clipped_any = False
    if x < 0 or x > IMG_W or y < 0 or y > IMG_H:
        clipped_any = True
    x = min(max(x, 0.0), IMG_W)
    y = min(max(y, 0.0), IMG_H)
    if clipped_any:
        stats.clipped_boxes += 1
    return x / IMG_W, y / IMG_H


def resolve_rgb_ir(img_dir: Path, stem: str) -> tuple[Path, Path]:
    rgb = img_dir / f"{stem}_co.png"
    ir = img_dir / f"{stem}_ir.png"
    if not rgb.exists():
        raise FileNotFoundError(f"missing RGB image for {stem}: {rgb}")
    if not ir.exists():
        raise FileNotFoundError(f"missing IR image for {stem}: {ir}")
    return rgb, ir


def convert_label(src_label: Path, dst_label: Path, stats: ConversionStats) -> None:
    lines_out = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        vals = line.strip().split()
        if len(vals) != 14:
            stats.invalid_boxes += 1
            continue
        raw_cls = int(float(vals[3]))
        class_name = RAW2NAME.get(raw_cls)
        if class_name is None:
            stats.invalid_boxes += 1
            continue
        xs = [float(x) for x in vals[6:10]]
        ys = [float(y) for y in vals[10:14]]
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            stats.invalid_boxes += 1
            continue
        points = []
        for x, y in zip(xs, ys):
            nx, ny = normalize_point(x, y, stats)
            points.extend([f"{nx:.6f}", f"{ny:.6f}"])
        cls_id = NAME2ID[class_name]
        lines_out.append(f"{cls_id} " + " ".join(points))
        stats.labels_count += 1
        stats.classes[class_name] += 1
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    if not lines_out:
        stats.empty_labels_count += 1


def build_split_map(ann_dir: Path, fold_id: str) -> dict[str, list[str]]:
    train_path = ann_dir / f"fold{fold_id}.txt"
    test_path = ann_dir / f"fold{fold_id}test.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"missing official fold file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"missing official fold file: {test_path}")
    train_ids = sorted(set(train_path.read_text(encoding="utf-8").split()))
    test_ids = sorted(set(test_path.read_text(encoding="utf-8").split()))
    return {
        "train": train_ids,
        "val": test_ids,
        "test": test_ids,
    }


def convert_fold(src_root: Path, dst_root: Path, fold_id: str, link_mode: str) -> None:
    ann_dir = src_root / "Annotations1024"
    img_dir = src_root / "Vehicules1024"
    split_map = build_split_map(ann_dir, fold_id)

    ensure_output_dirs(dst_root)
    stats = ConversionStats()
    for split, stems in split_map.items():
        for stem in stems:
            rgb_src, ir_src = resolve_rgb_ir(img_dir, stem)
            materialize_file(rgb_src, dst_root / "images" / split / f"{stem}.png", link_mode)
            materialize_file(ir_src, dst_root / "images_ir" / split / f"{stem}.png", link_mode)
            label_path = dst_root / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            convert_label(ann_dir / f"{stem}.txt", label_path, stats)
            stats.images_count += 1

    write_multimodal_yaml(dst_root / "VEDAI_student_obb.yaml", dst_root, 6, "images", CLASS_NAMES)
    write_multimodal_yaml(dst_root / "VEDAI_teacher_rgb_obb.yaml", dst_root, 3, "images", CLASS_NAMES)
    write_multimodal_yaml(dst_root / "VEDAI_teacher_ir_obb.yaml", dst_root, 3, "images_ir", CLASS_NAMES)
    summary_lines = stats.format_lines(
        f"VEDAI YOLO OBB official CV conversion summary: fold{fold_id}",
        extra_lines=[
            f"source: {src_root}",
            f"destination: {dst_root}",
            "source kind: raw_rotated",
            f"official fold: fold{fold_id}",
            f"link mode: {link_mode}",
            "val split: official foldXXtest",
            "test split: official foldXXtest",
        ],
    )
    write_summary(dst_root / "README_conversion.txt", summary_lines)
    print("\n".join(summary_lines))


def main() -> None:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    fold_ids = parse_fold_ids(args.folds)

    if args.force and dst_root.exists():
        shutil.rmtree(dst_root)
    if dst_root.exists():
        raise FileExistsError(f"destination already exists: {dst_root}. Use --force to replace it.")

    dst_root.mkdir(parents=True, exist_ok=True)
    for fold_id in fold_ids:
        fold_root = dst_root / f"fold{fold_id}"
        convert_fold(src_root, fold_root, fold_id, args.link_mode)
        print("")


if __name__ == "__main__":
    main()
