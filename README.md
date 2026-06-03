# SEMD: Selective Entropy-guided Multimodal Distillation

SEMD is a PyTorch implementation for RGB-infrared oriented object detection.
It trains a compact dual-stream student detector with two single-modality
teachers and an entropy-aware fusion gate.

The release is intentionally lightweight. It contains the code, model
definitions, dataset YAML examples, conversion utilities, and reproducible
training commands. It does not include datasets, trained checkpoints, logs, or
large experiment artifacts.

## Highlights

- RGB-IR dual-stream student detector for OBB detection.
- Multi-scale `EntropyOffsetGateFusion` with offset alignment, spatial gate,
  channel gate, and cached gate probabilities.
- Dual-teacher supervision from RGB and IR teacher detectors.
- Gate KD from teacher confidence-derived modality preference.
- Student gate entropy weighting for uncertain fusion regions.
- Compact qualitative visualization tools for detection, modality preference,
  and entropy maps.

## Installation

Python 3.10 and PyTorch 2.1 or newer are recommended.

```bash
conda create -n semd python=3.10 -y
conda activate semd
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Dataset Layout

Prepare paired RGB and infrared images outside the repository, or under the
ignored `datasets/` directory. The default DroneVehicle-style OBB YAML is:

```text
data/DroneVehicle_obb_student_external.yaml
```

Expected layout:

```text
datasets/DroneVehicle_rgbir_obb/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── images_ir/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Labels use YOLO OBB format:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

VEDAI and LLVIP conversion helpers are provided in `tools/`.

## DroneVehicle Refined Annotations

This release includes SEMD refined annotations for DroneVehicle, but it does
not redistribute the original DroneVehicle images.

Use the official DroneVehicle page to obtain the original RGB-IR image pairs:

- https://github.com/VisDrone/DroneVehicle
- https://arxiv.org/abs/2003.02437

Provided annotation artifacts:

```text
annotations/dronevehicle_refined_yolo_obb/labels/        # shared YOLO-OBB labels for SEMD
annotations/dronevehicle_refined_coco/                   # cleaned modality-specific COCO JSON
annotations/dronevehicle_refined_yolo_obb/annotation_changes.csv  # object-level original-to-refined changes
DATASET.md
CHANGELOG_ANNOTATIONS.md
```

The refined annotations are built on top of DroneVehicle. Please cite the
original DroneVehicle paper when using this annotation release.

Install the refined YOLO-OBB labels into a local DroneVehicle-style dataset:

```bash
python tools/prepare_dronevehicle_refined_annotations.py \
  --dataset-root datasets/DroneVehicle_rgbir_obb \
  --overwrite \
  --write-yaml data/DroneVehicle_refined_obb.yaml
```

Validate the label format and, if local images are available, RGB/IR image
counts:

```bash
python tools/check_dronevehicle_refined_annotations.py \
  --label-root datasets/DroneVehicle_rgbir_obb/labels \
  --dataset-root datasets/DroneVehicle_rgbir_obb \
  --allow-empty
```

See `DATASET.md` for dataset attribution, release statistics, and citation
details.

## Teacher Checkpoints

Teacher checkpoints are not distributed with this repository. Place them at:

```text
weight/teacher_hgnetv2_obb_rgb/best.pt
weight/teacher_hgnetv2_obb_ir/best.pt
```

The `weight/**/README.md` placeholders are kept so the expected layout is
visible while `.pt` files remain ignored by Git.

## Train SEMD

Use the provided script:

```bash
bash docs/semd_train_command.sh
```

or call the trainer directly:

```bash
python train_student_deimhgnetv2_obb.py \
  --data data/DroneVehicle_obb_student_external.yaml \
  --model model_yaml_obb/yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml \
  --teacher-rgb weight/teacher_hgnetv2_obb_rgb/best.pt \
  --teacher-ir weight/teacher_hgnetv2_obb_ir/best.pt \
  --distill-cross-attention \
  --distill-normal-distillation \
  --distill-head-kd-policy off \
  --distill-disable-pseudo-fusion-kd \
  --distill-gate-kd-mode legacy \
  --distill-gate-kd-weight 1.0 \
  --distill-student-entropy-weight-enable \
  --distill-student-entropy-weight-mode fixed \
  --distill-student-entropy-weight-formula asym_centered \
  --distill-student-entropy-weight-target gate
```

The default recipe uses branch feature distillation, cross-modal distillation,
Gate KD, and student entropy weighting. Detection-head KD and pseudo-fusion KD
are disabled in the default release command.

## Evaluate

```bash
MODEL_PATH=runs/obb/semd_default/weights/best.pt bash docs/semd_test_command.sh
```

## Qualitative Figures

After preparing a SEMD checkpoint and optional teacher checkpoints:

```bash
python scripts/generate_semd_qualitative.py \
  --samples 07198,06197,04744,03374 \
  --out outputs/semd_qualitative \
  --device 0 \
  --paper-compact
```

The script can export detection comparisons, student/teacher modality
preference maps, and student gate entropy figures.

## Repository Layout

```text
.
├── data/                 # Dataset YAML templates
├── annotations/          # SEMD refined DroneVehicle annotations
├── docs/                 # Method summary and runnable commands
├── model_yaml_obb/       # SEMD OBB and teacher model definitions
├── scripts/              # Training wrappers and qualitative visualization
├── tools/                # Dataset conversion and metric helpers
├── ultralytics/          # Modified detection framework used by SEMD
├── train_student_deimhgnetv2_obb.py
└── train_teacher_hgnetv2_obb.py
```

## License

This codebase is derived from a modified Ultralytics-based framework and keeps
the AGPL-3.0 license designation. Review the license file before public
redistribution or commercial use.

## Citation

If this repository helps your research, please cite the SEMD paper or project
once the formal citation is available.

If you use the DroneVehicle annotations, also cite the original DroneVehicle
paper:

```bibtex
@ARTICLE{sun2020drone,
  title={Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3168279}
}
```
