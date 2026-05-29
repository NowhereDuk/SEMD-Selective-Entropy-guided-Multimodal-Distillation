#!/usr/bin/env python3
"""Convert VEDAI annotations into paired YOLO HBB multimodal layout."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Convert VEDAI to paired YOLO HBB layout.")
    parser.add_argument("--src", default="datasets/raw/VEDAI", help="VEDAI raw or COCO-style root.")
    parser.add_argument("--dst", default="datasets/VEDAI_hbb", help="Output dataset root.")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize paired images in the output dataset.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio from non-test images for raw VEDAI.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for raw VEDAI split.")
    parser.add_argument("--fold-train", default="fold01.txt", help="Official raw VEDAI train fold filename.")
    parser.add_argument("--fold-test", default="fold01test.txt", help="Official raw VEDAI test fold filename.")
    parser.add_argument("--force", action="store_true", help="Replace destination if it already exists.")
    return parser.parse_args()


def normalize_hbb(xmin, ymin, xmax, ymax, width, height, stats: ConversionStats):
    clipped_any = False
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        clipped_any = True
    xmin = min(max(xmin, 0.0), width)
    ymin = min(max(ymin, 0.0), height)
    xmax = min(max(xmax, 0.0), width)
    ymax = min(max(ymax, 0.0), height)
    if clipped_any:
        stats.clipped_boxes += 1
    if xmax <= xmin or ymax <= ymin:
        stats.invalid_boxes += 1
        return None
    return (
        ((xmin + xmax) * 0.5) / width,
        ((ymin + ymax) * 0.5) / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def resolve_rgb_ir_raw(img_dir: Path, stem: str) -> tuple[Path, Path]:
    rgb = img_dir / f"{stem}_co.png"
    ir = img_dir / f"{stem}_ir.png"
    if not rgb.exists():
        raise FileNotFoundError(f"missing RGB image for {stem}: {rgb}")
    if not ir.exists():
        raise FileNotFoundError(f"missing IR image for {stem}: {ir}")
    return rgb, ir


def convert_raw_label_to_hbb(src_label: Path, dst_label: Path, stats: ConversionStats) -> None:
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
        hbb = normalize_hbb(min(xs), min(ys), max(xs), max(ys), IMG_W, IMG_H, stats)
        if hbb is None:
            continue
        cls_id = NAME2ID[class_name]
        lines_out.append(f"{cls_id} {hbb[0]:.6f} {hbb[1]:.6f} {hbb[2]:.6f} {hbb[3]:.6f}")
        stats.labels_count += 1
        stats.classes[class_name] += 1
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    if not lines_out:
        stats.empty_labels_count += 1


def convert_raw_vedai(src_root: Path, dst_root: Path, args: argparse.Namespace) -> ConversionStats:
    ann_dir = src_root / "Annotations1024"
    img_dir = src_root / "Vehicules1024"
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
            rgb_src, ir_src = resolve_rgb_ir_raw(img_dir, stem)
            materialize_file(rgb_src, dst_root / "images" / split / f"{stem}.png", args.link_mode)
            materialize_file(ir_src, dst_root / "images_ir" / split / f"{stem}.png", args.link_mode)
            label_path = dst_root / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            convert_raw_label_to_hbb(ann_dir / f"{stem}.txt", label_path, stats)
            stats.images_count += 1
    return stats


def resolve_coco_pair(split_dir: Path, stem: str) -> tuple[Path, Path]:
    rgb = split_dir / f"{stem}_co.png"
    ir = split_dir / f"{stem}_ir.png"
    if not rgb.exists():
        raise FileNotFoundError(f"missing RGB image for {stem}: {rgb}")
    if not ir.exists():
        raise FileNotFoundError(f"missing IR image for {stem}: {ir}")
    return rgb, ir


def convert_coco_vedai(src_root: Path, dst_root: Path, args: argparse.Namespace) -> ConversionStats:
    stats = ConversionStats()
    ann_root = src_root / "annotations"
    img_root = src_root / "images"
    for split in ("train", "val", "test"):
        ann_path = ann_root / f"instances_{split}.json"
        if not ann_path.exists():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        images = {item["id"]: item for item in data.get("images", [])}
        anns_by_image = {}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        cat_id_to_name = {item["id"]: item["name"] for item in data.get("categories", [])}
        for image_id, image_info in images.items():
            stem = Path(image_info["file_name"]).stem.replace("_co", "").replace("_ir", "")
            rgb_src, ir_src = resolve_coco_pair(img_root / split, stem)
            materialize_file(rgb_src, dst_root / "images" / split / rgb_src.name, args.link_mode)
            materialize_file(ir_src, dst_root / "images_ir" / split / ir_src.name, args.link_mode)
            width = float(image_info.get("width", IMG_W))
            height = float(image_info.get("height", IMG_H))
            lines_out = []
            for ann in anns_by_image.get(image_id, []):
                class_name = cat_id_to_name.get(ann["category_id"], "")
                if class_name not in NAME2ID:
                    stats.invalid_boxes += 1
                    continue
                xmin, ymin, box_w, box_h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
                hbb = normalize_hbb(xmin, ymin, xmin + box_w, ymin + box_h, width, height, stats)
                if hbb is None:
                    continue
                cls_id = NAME2ID[class_name]
                lines_out.append(f"{cls_id} {hbb[0]:.6f} {hbb[1]:.6f} {hbb[2]:.6f} {hbb[3]:.6f}")
                stats.labels_count += 1
                stats.classes[class_name] += 1
            label_path = dst_root / "labels" / split / f"{stem}.txt"
            label_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
            if not lines_out:
                stats.empty_labels_count += 1
            stats.images_count += 1
    return stats


def main() -> None:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if args.force and dst_root.exists():
        shutil.rmtree(dst_root)
    if dst_root.exists():
        raise FileExistsError(f"destination already exists: {dst_root}. Use --force to replace it.")

    ensure_output_dirs(dst_root)
    if (src_root / "annotations").exists():
        source_kind = "coco_hbb"
        stats = convert_coco_vedai(src_root, dst_root, args)
    else:
        source_kind = "raw_rotated"
        stats = convert_raw_vedai(src_root, dst_root, args)

    write_multimodal_yaml(dst_root / "VEDAI_student_hbb.yaml", dst_root, 6, "images", CLASS_NAMES)
    write_multimodal_yaml(dst_root / "VEDAI_teacher_rgb_hbb.yaml", dst_root, 3, "images", CLASS_NAMES)
    write_multimodal_yaml(dst_root / "VEDAI_teacher_ir_hbb.yaml", dst_root, 3, "images_ir", CLASS_NAMES)
    summary_lines = stats.format_lines(
        "VEDAI YOLO HBB conversion summary",
        extra_lines=[
            f"source: {src_root}",
            f"destination: {dst_root}",
            f"source kind: {source_kind}",
            f"link mode: {args.link_mode}",
        ],
    )
    write_summary(dst_root / "README_conversion.txt", summary_lines)
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
