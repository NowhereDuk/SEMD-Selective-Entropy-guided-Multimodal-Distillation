#!/usr/bin/env python3
"""Shared helpers for multimodal YOLO dataset conversion scripts."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConversionStats:
    images_count: int = 0
    labels_count: int = 0
    empty_labels_count: int = 0
    invalid_boxes: int = 0
    clipped_boxes: int = 0
    classes: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "images count": self.images_count,
            "labels count": self.labels_count,
            "empty labels count": self.empty_labels_count,
            "classes": dict(sorted(self.classes.items())),
            "invalid boxes": self.invalid_boxes,
            "clipped boxes": self.clipped_boxes,
        }

    def format_lines(self, title: str, extra_lines: list[str] | None = None) -> list[str]:
        lines = [title]
        if extra_lines:
            lines.extend(extra_lines)
        summary = self.to_dict()
        lines.extend([
            f"images count: {summary['images count']}",
            f"labels count: {summary['labels count']}",
            f"empty labels count: {summary['empty labels count']}",
            f"classes: {json.dumps(summary['classes'], ensure_ascii=True, sort_keys=True)}",
            f"invalid boxes: {summary['invalid boxes']}",
            f"clipped boxes: {summary['clipped boxes']}",
        ])
        return lines


def safe_unlink(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    safe_unlink(dst)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            os.symlink(src, dst)
            return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    shutil.copy2(src, dst)


def ensure_output_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "images_ir" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def clip(value: float, low: float, high: float) -> tuple[float, bool]:
    clipped = min(max(value, low), high)
    return clipped, clipped != value


def write_multimodal_yaml(path: Path, dataset_root: Path, ch: int, image_root: str, class_names: list[str]) -> None:
    lines = [
        f"path: {dataset_root}",
        "",
        f"train: {image_root}/train",
        f"val: {image_root}/val",
        f"test: {image_root}/test",
        "",
        f"nc: {len(class_names)}",
        f"ch: {ch}",
        "",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
