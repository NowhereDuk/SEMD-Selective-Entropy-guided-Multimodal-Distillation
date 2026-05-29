#!/usr/bin/env python3
"""Convert raw VEDAI rotated annotations into paired YOLO OBB multimodal layout."""

from __future__ import annotations

import argparse
import random
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
    parser = argparse.ArgumentParser(description="Convert raw VEDAI to paired YOLO OBB layout.")
    parser.add_argument("--src", default="datasets/raw/VEDAI", help="Raw VEDAI root.")
    parser.add_argument("--dst", default="datasets/VEDAI_obb", help="Output dataset root.")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize paired images in the output dataset.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio from non-test images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for validation split.")
    parser.add_argument("--fold-train", default="fold01.txt", help="Official raw VEDAI train fold filename.")
    parser.add_argument("--fold-test", default="fold01test.txt", help="Official raw VEDAI test fold filename.")
    parser.add_argument("--force", action="store_true", help="Replace destination if it already exists.")
    return parser.parse_args()


def normalize_point(x, y, stats: ConversionStats):
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


def main() -> None:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    ann_dir = src_root / "Annotations1024"
    img_dir = src_root / "Vehicules1024"

    if args.force and dst_root.exists():
        shutil.rmtree(dst_root)
    if dst_root.exists():
        raise FileExistsError(f"destination already exists: {dst_root}. Use --force to replace it.")

    ensure_output_dirs(dst_root)
    train_ids = set((ann_dir / args.fold_train).read_text(encoding="utf-8").split())
    test_ids = set((ann_dir / args.fold_test).read_text(encoding="utf-8").split())
    annotated_ids = sorted(p.stem for p in ann_dir.glob("*.txt") if p.stem.isdigit())
    trainval_ids = sorted((set(annotated_ids) - test_ids) | (train_ids - test_ids))
    rng = random.Random(args.seed)
    rng.shuffle(trainval_ids)
    val_count = max(1, int(len(trainval_ids) * args.val_ratio))
    val_ids = set(trainval_ids[:val_count])
    split_map = {
        "train": sorted(set(trainval_ids[val_count:])),
        "val": sorted(val_ids),
        "test": sorted(set(annotated_ids) & test_ids),
    }

    stats = ConversionStats()
    for split, stems in split_map.items():
        for stem in stems:
            rgb_src, ir_src = resolve_rgb_ir(img_dir, stem)
            materialize_file(rgb_src, dst_root / "images" / split / f"{stem}.png", args.link_mode)
            materialize_file(ir_src, dst_root / "images_ir" / split / f"{stem}.png", args.link_mode)
            label_path = dst_root / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            convert_label(ann_dir / f"{stem}.txt", label_path, stats)
            stats.images_count += 1

    write_multimodal_yaml(dst_root / "VEDAI_student_obb.yaml", dst_root, 6, "images", CLASS_NAMES)
    write_multimodal_yaml(dst_root / "VEDAI_teacher_rgb_obb.yaml", dst_root, 3, "images", CLASS_NAMES)
    write_multimodal_yaml(dst_root / "VEDAI_teacher_ir_obb.yaml", dst_root, 3, "images_ir", CLASS_NAMES)
    summary_lines = stats.format_lines(
        "VEDAI YOLO OBB conversion summary",
        extra_lines=[
            f"source: {src_root}",
            f"destination: {dst_root}",
            "source kind: raw_rotated",
            f"link mode: {args.link_mode}",
        ],
    )
    write_summary(dst_root / "README_conversion.txt", summary_lines)
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
