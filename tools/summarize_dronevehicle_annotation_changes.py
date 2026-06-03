#!/usr/bin/env python3
"""Aggregate instance-level DroneVehicle refinement changes to file-level CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("/home/disk1/DataSets/DroneVehicle_adjust/Annotation_cleaned/change_list_from_original/change_list.csv")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "annotations"
    / "dronevehicle_refined_yolo_obb"
    / "annotation_changes.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Instance-level change_list.csv.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="File-level output CSV.")
    return parser.parse_args()


def classify_change(change_type: str) -> str:
    text = change_type.lower()
    if "addition" in text:
        return "added"
    if "removal" in text:
        return "removed"
    return "modified"


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing instance change list: {args.input}")

    grouped = defaultdict(
        lambda: {
            "added": 0,
            "removed": 0,
            "modified": 0,
            "modalities": set(),
            "classes": set(),
            "change_types": Counter(),
            "verification": Counter(),
        }
    )

    with args.input.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "image_id", "modality", "original_category", "refined_category", "change_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{args.input} missing columns: {sorted(missing)}")

        for row in reader:
            key = (row["split"], row["image_id"])
            item = grouped[key]
            item[classify_change(row["change_type"])] += 1
            item["modalities"].add(row["modality"])
            item["change_types"][row["change_type"]] += 1
            item["verification"][row.get("verification_status", "")] += 1
            for col in ("original_category", "refined_category"):
                value = row.get(col, "")
                if value and value != "none":
                    item["classes"].add(value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        fieldnames = [
            "split",
            "image_id",
            "label_file",
            "modalities",
            "added",
            "removed",
            "modified",
            "classes_affected",
            "change_types",
            "verification_statuses",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split, image_id in sorted(grouped):
            item = grouped[(split, image_id)]
            change_types = ";".join(f"{k}:{v}" for k, v in sorted(item["change_types"].items()))
            verification = ";".join(f"{k}:{v}" for k, v in sorted(item["verification"].items()) if k)
            writer.writerow(
                {
                    "split": split,
                    "image_id": image_id,
                    "label_file": f"{split}/{image_id}.txt",
                    "modalities": "|".join(sorted(item["modalities"])),
                    "added": item["added"],
                    "removed": item["removed"],
                    "modified": item["modified"],
                    "classes_affected": "|".join(sorted(item["classes"])),
                    "change_types": change_types,
                    "verification_statuses": verification,
                    "note": "file-level summary aggregated from instance-level refinement records",
                }
            )

    print(f"Wrote {len(grouped)} file-level rows to {args.output}")


if __name__ == "__main__":
    main()
