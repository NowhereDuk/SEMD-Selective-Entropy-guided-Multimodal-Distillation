# SEMD Refined DroneVehicle YOLO-OBB Labels

This directory contains shared YOLO-OBB labels for RGB-IR SEMD student training
on DroneVehicle. It does not contain the original DroneVehicle images.

## Layout

```text
labels/
├── train/
├── val/
└── test/
annotation_changes.csv
change_summary.json
label_stats.json
```

`annotation_changes.csv` is the object-level original-to-refined change list.
It has one row per detected annotation change and uses the Table A2 fields:

```text
change_id,split,image_id,modality,object_id,original_category,refined_category,original_obb,refined_obb,change_type,match_iou,match_confidence,verification_status
```

Each label row has normalized polygon coordinates:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

Class mapping:

```text
0 car
1 freight_car
2 truck
3 bus
4 van
```

## Install Into a Local Dataset

```bash
python tools/prepare_dronevehicle_refined_annotations.py \
  --dataset-root datasets/DroneVehicle_rgbir_obb \
  --overwrite \
  --write-yaml data/DroneVehicle_refined_obb.yaml
```

Use `--mode symlink` instead of the default copy mode if you want the local
dataset to reference the labels in this repository.

## Validate

```bash
python tools/check_dronevehicle_refined_annotations.py \
  --label-root annotations/dronevehicle_refined_yolo_obb/labels \
  --allow-empty
```

With local RGB and IR images:

```bash
python tools/check_dronevehicle_refined_annotations.py \
  --label-root datasets/DroneVehicle_rgbir_obb/labels \
  --dataset-root datasets/DroneVehicle_rgbir_obb \
  --allow-empty \
  --visualize 8
```

Empty label files are expected for images with no retained objects after
cleaning.
