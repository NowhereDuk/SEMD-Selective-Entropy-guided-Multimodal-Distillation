#!/usr/bin/env python3
"""Export the object-level DroneVehicle original-to-refined change list."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("/home/disk1/DataSets/DroneVehicle_adjust/Annotation_cleaned/change_list_from_original/change_list.csv")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "annotations"
    / "dronevehicle_refined_yolo_obb"
    / "annotation_changes.csv"
)
TABLE_A2_FIELDS = [
    "change_id",
    "split",
    "image_id",
    "modality",
    "object_id",
    "original_category",
    "refined_category",
    "original_obb",
    "refined_obb",
    "change_type",
    "match_iou",
    "match_confidence",
    "verification_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Full object-level change_list.csv.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Table A2-compatible output CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing object-level change list: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.input.open(newline="") as src, args.output.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        missing = set(TABLE_A2_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{args.input} missing columns: {sorted(missing)}")

        writer = csv.DictWriter(dst, fieldnames=TABLE_A2_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            writer.writerow({field: row.get(field, "") for field in TABLE_A2_FIELDS})
            rows += 1

    print(f"Wrote {rows} object-level rows to {args.output}")


if __name__ == "__main__":
    main()
