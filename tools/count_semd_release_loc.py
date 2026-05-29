#!/usr/bin/env python3
"""Count SEMD release LOC while excluding unmodified third-party framework code."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_SELF = "tools/count_semd_release_loc.py"

STANDALONE_FILES = [
    "data/DroneVehicle_obb_student_external.yaml",
    "data/LLVIP_hbb_student_external.yaml",
    "data/VEDAI_hbb_student_external.yaml",
    "data/VEDAI_obb_student_external.yaml",
    "docs/semd_test_command.sh",
    "docs/semd_train_command.sh",
    "model_yaml_hbb/yolov8-EntropyOffsetGate-deimhgnetv2-b0.yaml",
    "model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml",
    "model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-p2-obb.yaml",
    "model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-p2p4-obb.yaml",
    "model_yaml_obb/yolov8-EntropyOffsetGateEfficient-deimhgnetv2-b0-obb.yaml",
    "model_yaml_obb/yolov8-EntropyOffsetGateLite-deimhgnetv2-b0-e2-obb.yaml",
    "model_yaml_obb/yolov8-hgnetv2-b0-obb-teacher.yaml",
    "model_yaml_obb/yolov8-hgnetv2-b0-teacher-obb.yaml",
    "scripts/generate_semd_qualitative.py",
    "scripts/train_student_deimhgnetv2_obb.py",
    "scripts/train_vedai_obb_300_bg.sh",
    "scripts/train_vedai_obb_cv10_300_bg.sh",
    "tools/convert_llvip_to_yolo_hbb.py",
    "tools/convert_vedai_to_yolo_hbb.py",
    "tools/convert_vedai_to_yolo_obb.py",
    "tools/convert_vedai_to_yolo_obb_cv.py",
    "tools/dataset_conversion_utils.py",
    "tools/eval_hbb_teacher_metrics.py",
    "train_student_deimhgnetv2_hbb.py",
    "train_student_deimhgnetv2_obb.py",
    "train_student_deimhgnetv2_obb_nodist.py",
    "train_teacher_hgnetv2_hbb.py",
    "train_teacher_hgnetv2_obb.py",
]

FRAMEWORK_RANGES = {
    "ultralytics/nn/modules/block.py": [
        ("entropy_fusion", "_normalize_spatial_map", "class MF"),
    ],
    "ultralytics/engine/trainer.py": [
        ("student_entropy_and_multimodal_distill", "class StudentEntropyWeightHead", "class BaseTrainer"),
    ],
    "ultralytics/nn/tasks.py": [
        ("model_parser_integration", "def parse_model", "def yaml_model_load"),
    ],
    "ultralytics/nn/modules/__init__.py": [
        ("module_exports", None, None),
    ],
}


def code_lines(lines: list[str]) -> int:
    return sum(1 for line in lines if line.strip() and not line.lstrip().startswith("#"))


def read_lines(rel: str) -> list[str]:
    return (ROOT / rel).read_text(encoding="utf-8", errors="ignore").splitlines()


def count_file(rel: str) -> int:
    if rel == EXCLUDED_SELF:
        return 0
    return code_lines(read_lines(rel))


def count_range(rel: str, start_marker: str | None, end_marker: str | None) -> int:
    lines = read_lines(rel)
    start = 0
    end = len(lines)
    if start_marker:
        start = next(i for i, line in enumerate(lines) if start_marker in line)
    if end_marker:
        end = next((i for i, line in enumerate(lines[start + 1 :], start + 1) if end_marker in line), len(lines))
    return code_lines(lines[start:end])


def main() -> None:
    standalone = [(rel, count_file(rel)) for rel in STANDALONE_FILES]
    framework = []
    for rel, ranges in FRAMEWORK_RANGES.items():
        for label, start_marker, end_marker in ranges:
            framework.append((f"{rel}:{label}", count_range(rel, start_marker, end_marker)))

    standalone_total = sum(count for _, count in standalone)
    framework_total = sum(count for _, count in framework)
    total = standalone_total + framework_total

    print("SEMD release independent code line statistics")
    print(f"Repository: {ROOT}")
    print()
    print("Counting rule:")
    print("  - Count non-empty, non-comment lines.")
    print("  - Exclude datasets, checkpoints, training outputs, caches, and unmodified third-party framework files.")
    print("  - Include SEMD training entries, configs, scripts, tools, and selected SEMD framework integration ranges.")
    print("  - Exclude this counting utility from the reported total.")
    print()
    print(f"Standalone SEMD release files:              {standalone_total:,} lines")
    print(f"Selected SEMD framework integration ranges: {framework_total:,} lines")
    print("-------------------------------------------------------")
    print(f"Independent developed code total:           {total:,} lines")


if __name__ == "__main__":
    main()
