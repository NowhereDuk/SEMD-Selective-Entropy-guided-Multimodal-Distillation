#!/usr/bin/env python3
"""Evaluate single-stream HBB teacher checkpoints and save compact metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--ch", type=int, default=3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--split", default="test")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs/detect_teacher_test"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weight = Path(args.weight).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    if not weight.exists():
        raise FileNotFoundError(f"checkpoint not found: {weight}")
    if not data.exists():
        raise FileNotFoundError(f"data yaml not found: {data}")

    model = YOLO(str(weight))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        ch=args.ch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name or args.label,
        exist_ok=True,
        iou=args.iou,
    )

    box = metrics.box
    summary = {
        "label": args.label,
        "weight": str(weight),
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "ch": args.ch,
        "iou": args.iou,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "map75": float(box.map75),
        "mp": float(box.mp),
        "mr": float(box.mr),
        "save_dir": str(metrics.save_dir),
    }

    json_out = Path(args.json_out).expanduser()
    if not json_out.is_absolute():
        json_out = REPO_ROOT / json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
