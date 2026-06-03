# Dataset and Refined DroneVehicle Annotations

This repository does not redistribute the original DroneVehicle images. It
provides SEMD refined annotations and helper tools for users who have obtained
DroneVehicle from the official source.

## Original DroneVehicle

Download the original DroneVehicle data from the official project page:

- Official repository: https://github.com/VisDrone/DroneVehicle
- Paper: https://arxiv.org/abs/2003.02437

The original DroneVehicle project describes the dataset as 28,439 paired
RGB-infrared samples, 56,878 images in total, with oriented bounding box
annotations for five vehicle categories. The official repository does not state
a broad redistribution license for mirroring the full image dataset here, so
users should follow the original download and usage terms.

## SEMD Refined Annotations

The refined annotations are built on top of DroneVehicle. Please cite the
original DroneVehicle paper when using this annotation release.

Provided formats:

- `annotations/dronevehicle_refined_yolo_obb/labels/`: shared YOLO-OBB labels
  for SEMD RGB-IR student training.
- `annotations/dronevehicle_refined_coco/`: modality-specific cleaned COCO JSON
  annotations derived from `/home/disk1/DataSets/DroneVehicle_adjust/Annotation_cleaned`.
- `annotations/dronevehicle_refined_yolo_obb/annotation_changes.csv`: object-level
  original-to-refined change list aligned with Table A2.

The YOLO-OBB class mapping is:

```text
0 car
1 freight_car
2 truck
3 bus
4 van
```

The YOLO-OBB line format is:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

All coordinates are normalized to `[0, 1]`.

## Usage

Prepare a local DroneVehicle-style RGB-IR dataset root:

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

Install the refined YOLO-OBB labels:

```bash
python tools/prepare_dronevehicle_refined_annotations.py \
  --dataset-root datasets/DroneVehicle_rgbir_obb \
  --overwrite \
  --write-yaml data/DroneVehicle_refined_obb.yaml
```

Validate the installed labels against local RGB and IR image counts:

```bash
python tools/check_dronevehicle_refined_annotations.py \
  --label-root datasets/DroneVehicle_rgbir_obb/labels \
  --dataset-root datasets/DroneVehicle_rgbir_obb \
  --allow-empty
```

Some DroneVehicle images have no retained object after cleaning, so empty label
files are expected and should be accepted with `--allow-empty`.

## Current Release Statistics

YOLO-OBB label files:

```text
train: 17,990
val:    1,469
test:   8,980
total: 28,439
```

YOLO-OBB objects:

```text
train: 316,022
val:    24,477
test:  159,453
total: 499,952
```

Empty label files:

```text
train: 41
val:    2
test:  23
total: 66
```

Object-level change list:

```text
annotation_changes.csv rows: 175,386
```

## Citation

Please cite the original DroneVehicle work when using this annotation release:

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
