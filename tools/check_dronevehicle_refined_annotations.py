#!/usr/bin/env python3
"""Validate and optionally visualize SEMD refined DroneVehicle YOLO-OBB labels."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - handled at runtime for visualization only
    Image = None
    ImageDraw = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_ROOT = REPO_ROOT / "annotations" / "dronevehicle_refined_yolo_obb" / "labels"
SPLITS = ("train", "val", "test")
CLASS_NAMES = {
    0: "car",
    1: "freight_car",
    2: "truck",
    3: "bus",
    4: "van",
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root containing images/ and images_ir/ for count checks and visualization.",
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--allow-empty", action="store_true", help="Allow empty label files.")
    parser.add_argument("--stats-json", type=Path, default=None, help="Optional JSON stats output path.")
    parser.add_argument("--visualize", type=int, default=0, help="Number of random samples to visualize.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dronevehicle_refined_label_vis"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def iter_images(image_dir: Path) -> Iterable[Path]:
    for ext in IMAGE_EXTS:
        yield from image_dir.glob(f"*{ext}")
        yield from image_dir.glob(f"*{ext.upper()}")


def find_image(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
        upper = image_dir / f"{stem}{ext.upper()}"
        if upper.exists():
            return upper
    return None


def parse_label_line(line: str, path: Path, line_no: int) -> Tuple[int, List[float]]:
    parts = line.split()
    if len(parts) != 9:
        raise ValueError(f"{path}:{line_no}: expected 9 columns, got {len(parts)}")
    try:
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: non-numeric value") from exc
    if cls not in CLASS_NAMES:
        raise ValueError(f"{path}:{line_no}: invalid class id {cls}")
    for value in coords:
        if not math.isfinite(value):
            raise ValueError(f"{path}:{line_no}: non-finite coordinate")
        if value < -1e-6 or value > 1.0 + 1e-6:
            raise ValueError(f"{path}:{line_no}: coordinate out of [0, 1]: {value}")
    return cls, coords


def validate_split(label_dir: Path, allow_empty: bool) -> dict:
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label split directory: {label_dir}")

    class_counts: Counter[int] = Counter()
    empty_files = 0
    objects = 0
    files = sorted(label_dir.glob("*.txt"))
    for path in files:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            empty_files += 1
            if not allow_empty:
                raise ValueError(f"{path}: empty label file")
        for line_no, line in enumerate(lines, start=1):
            cls, _ = parse_label_line(line, path, line_no)
            class_counts[cls] += 1
            objects += 1

    return {
        "label_files": len(files),
        "objects": objects,
        "empty_files": empty_files,
        "class_counts": {CLASS_NAMES[k]: class_counts[k] for k in sorted(CLASS_NAMES)},
    }


def check_image_counts(dataset_root: Path, split: str, label_stems: set[str]) -> dict:
    rgb_dir = dataset_root / "images" / split
    ir_dir = dataset_root / "images_ir" / split
    result = {"rgb_images": None, "ir_images": None, "count_match": None, "stem_match": None}
    rgb_stems = None
    ir_stems = None
    if rgb_dir.is_dir():
        rgb_stems = {path.stem for path in iter_images(rgb_dir)}
        result["rgb_images"] = len(rgb_stems)
    if ir_dir.is_dir():
        ir_stems = {path.stem for path in iter_images(ir_dir)}
        result["ir_images"] = len(ir_stems)
    if rgb_stems is not None and ir_stems is not None:
        result["count_match"] = result["rgb_images"] == result["ir_images"] == len(label_stems)
        result["stem_match"] = rgb_stems == ir_stems == label_stems
        if not result["count_match"]:
            raise ValueError(
                f"{split}: labels={len(label_stems)}, rgb={result['rgb_images']}, ir={result['ir_images']}"
            )
        if not result["stem_match"]:
            missing_rgb = sorted(label_stems - rgb_stems)[:5]
            missing_ir = sorted(label_stems - ir_stems)[:5]
            extra_rgb = sorted(rgb_stems - label_stems)[:5]
            extra_ir = sorted(ir_stems - label_stems)[:5]
            raise ValueError(
                f"{split}: image/label stems do not match; "
                f"missing_rgb={missing_rgb}, missing_ir={missing_ir}, "
                f"extra_rgb={extra_rgb}, extra_ir={extra_ir}"
            )
    return result


def draw_label_panel(image_path: Path, label_path: Path):
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for --visualize")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        cls, coords = parse_label_line(line.strip(), label_path, line_no)
        pts = [(coords[i] * width, coords[i + 1] * height) for i in range(0, 8, 2)]
        draw.line(pts + [pts[0]], fill=(255, 64, 64), width=2)
        draw.text(pts[0], CLASS_NAMES[cls], fill=(255, 255, 0))
    return image


def visualize_samples(label_root: Path, dataset_root: Path, splits: List[str], output_dir: Path, n: int, seed: int) -> None:
    candidates = []
    for split in splits:
        for label_path in sorted((label_root / split).glob("*.txt")):
            stem = label_path.stem
            rgb_path = find_image(dataset_root / "images" / split, stem)
            ir_path = find_image(dataset_root / "images_ir" / split, stem)
            if rgb_path is not None and ir_path is not None:
                candidates.append((split, label_path, rgb_path, ir_path))
    if not candidates:
        raise FileNotFoundError("No image/label pairs found for visualization")

    random.Random(seed).shuffle(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, label_path, rgb_path, ir_path in candidates[:n]:
        rgb = draw_label_panel(rgb_path, label_path)
        ir = draw_label_panel(ir_path, label_path)
        canvas = Image.new("RGB", (rgb.width + ir.width, max(rgb.height, ir.height)), "white")
        canvas.paste(rgb, (0, 0))
        canvas.paste(ir, (rgb.width, 0))
        canvas.save(output_dir / f"{split}_{label_path.stem}.jpg", quality=95)


def main() -> None:
    args = parse_args()
    stats = {"splits": {}, "total": defaultdict(int)}

    for split in args.splits:
        split_stats = validate_split(args.label_root / split, args.allow_empty)
        if args.dataset_root is not None:
            label_stems = {path.stem for path in (args.label_root / split).glob("*.txt")}
            split_stats["images"] = check_image_counts(args.dataset_root, split, label_stems)
        stats["splits"][split] = split_stats
        stats["total"]["label_files"] += split_stats["label_files"]
        stats["total"]["objects"] += split_stats["objects"]
        stats["total"]["empty_files"] += split_stats["empty_files"]

    stats["total"] = dict(stats["total"])
    print(json.dumps(stats, indent=2, sort_keys=True))

    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    if args.visualize > 0:
        if args.dataset_root is None:
            raise ValueError("--visualize requires --dataset-root")
        visualize_samples(args.label_root, args.dataset_root, args.splits, args.output_dir, args.visualize, args.seed)


if __name__ == "__main__":
    main()
