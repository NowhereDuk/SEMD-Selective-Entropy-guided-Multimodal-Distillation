#!/usr/bin/env python3
"""Convert raw LLVIP XML annotations into paired YOLO HBB multimodal layout."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from dataset_conversion_utils import ConversionStats, ensure_output_dirs, materialize_file, write_multimodal_yaml, write_summary


CLASS_NAMES = ["person"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw LLVIP to paired YOLO HBB layout.")
    parser.add_argument("--src", default="datasets/raw/LLVIP", help="Raw LLVIP root.")
    parser.add_argument("--dst", default="datasets/LLVIP_hbb", help="Output dataset root.")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize paired images in the output dataset.",
    )
    parser.add_argument("--force", action="store_true", help="Replace destination if it already exists.")
    return parser.parse_args()


def normalize_box(xmin, ymin, xmax, ymax, width, height, stats: ConversionStats):
    clipped_any = False
    xmin, changed = max(float(xmin), 0.0), float(xmin) < 0.0
    clipped_any |= changed
    ymin, changed = max(float(ymin), 0.0), float(ymin) < 0.0
    clipped_any |= changed
    xmax, changed = min(float(xmax), width), float(xmax) > width
    clipped_any |= changed
    ymax, changed = min(float(ymax), height), float(ymax) > height
    clipped_any |= changed
    if clipped_any:
        stats.clipped_boxes += 1
    if xmax <= xmin or ymax <= ymin:
        stats.invalid_boxes += 1
        return None
    x_center = ((xmin + xmax) * 0.5) / width
    y_center = ((ymin + ymax) * 0.5) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return x_center, y_center, box_w, box_h


def convert_xml_to_yolo(xml_path: Path, label_path: Path, stats: ConversionStats) -> int:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing <size> in {xml_path}")
    width = float(size.findtext("width", default="0"))
    height = float(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {xml_path}: {width}x{height}")

    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name not in CLASS_TO_ID:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            stats.invalid_boxes += 1
            continue
        box = normalize_box(
            bbox.findtext("xmin", default="0"),
            bbox.findtext("ymin", default="0"),
            bbox.findtext("xmax", default="0"),
            bbox.findtext("ymax", default="0"),
            width,
            height,
            stats,
        )
        if box is None:
            continue
        cls_id = CLASS_TO_ID[name]
        stats.classes[name] += 1
        stats.labels_count += 1
        lines.append(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if not lines:
        stats.empty_labels_count += 1
    return len(lines)


def raw_to_split(raw_split: str) -> str:
    return "val" if raw_split == "test" else raw_split


def main() -> None:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    ann_dir = src_root / "Annotations"

    if args.force and dst_root.exists():
        shutil.rmtree(dst_root)
    if dst_root.exists():
        raise FileExistsError(f"destination already exists: {dst_root}. Use --force to replace it.")

    ensure_output_dirs(dst_root)
    stats = ConversionStats()

    for raw_split in ("train", "test"):
        split = raw_to_split(raw_split)
        rgb_dir = src_root / "visible" / raw_split
        ir_dir = src_root / "infrared" / raw_split
        rgb_stems = {p.stem for p in rgb_dir.glob("*.jpg")}
        ir_stems = {p.stem for p in ir_dir.glob("*.jpg")}
        for stem in sorted(rgb_stems & ir_stems):
            rgb_src = rgb_dir / f"{stem}.jpg"
            ir_src = ir_dir / f"{stem}.jpg"
            xml_src = ann_dir / f"{stem}.xml"
            if not xml_src.exists():
                raise FileNotFoundError(f"missing annotation for {stem}: {xml_src}")
            materialize_file(rgb_src, dst_root / "images" / split / f"{stem}.jpg", args.link_mode)
            materialize_file(ir_src, dst_root / "images_ir" / split / f"{stem}.jpg", args.link_mode)
            label_path = dst_root / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            convert_xml_to_yolo(xml_src, label_path, stats)
            stats.images_count += 1

    write_multimodal_yaml(dst_root / "LLVIP_student.yaml", dst_root, 6, "images", CLASS_NAMES)
    write_multimodal_yaml(dst_root / "LLVIP_teacher_rgb.yaml", dst_root, 3, "images", CLASS_NAMES)
    write_multimodal_yaml(dst_root / "LLVIP_teacher_ir.yaml", dst_root, 3, "images_ir", CLASS_NAMES)
    summary_lines = stats.format_lines(
        "LLVIP YOLO HBB conversion summary",
        extra_lines=[
            f"source: {src_root}",
            f"destination: {dst_root}",
            "split mapping: raw train -> train, raw test -> val",
            f"link mode: {args.link_mode}",
        ],
    )
    write_summary(dst_root / "README_conversion.txt", summary_lines)
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
