# DroneVehicle Annotation Changelog

This file summarizes the SEMD refined DroneVehicle annotation release. The
machine-readable object-level original-to-refined change list is:

```text
annotations/dronevehicle_refined_yolo_obb/annotation_changes.csv
```

The refined annotations are derived from:

```text
/home/disk1/DataSets/DroneVehicle_adjust/Annotation_cleaned
```

## Release Contents

- YOLO-OBB labels: `annotations/dronevehicle_refined_yolo_obb/labels/`
- Cleaned COCO JSON: `annotations/dronevehicle_refined_coco/`
- Object-level change CSV: `annotations/dronevehicle_refined_yolo_obb/annotation_changes.csv`
- Cleaning summary JSON: `annotations/dronevehicle_refined_coco/cleaning_summary.json`
- Change summary JSON: `annotations/dronevehicle_refined_yolo_obb/change_summary.json`

## Object-Level Change List

`annotation_changes.csv` uses one row per detected annotation change and keeps
the fields reported in Table A2:

```text
change_id,split,image_id,modality,object_id,original_category,refined_category,original_obb,refined_obb,change_type,match_iou,match_confidence,verification_status
```

The CSV has 175,386 data rows. It is exported from the full object-level
cleaning records while omitting auxiliary internal columns not listed in Table
A2.

## Label Statistics

```text
train labels: 17,990 files, 316,022 objects, 41 empty files
val labels:    1,469 files,  24,477 objects,  2 empty files
test labels:   8,980 files, 159,453 objects, 23 empty files
total:        28,439 files, 499,952 objects, 66 empty files
```

Class counts are recorded in:

```text
annotations/dronevehicle_refined_yolo_obb/label_stats.json
```

## Regeneration

To regenerate the object-level CSV from the full cleaning source on the release
machine:

```bash
python tools/export_dronevehicle_annotation_changes.py \
  --input /home/disk1/DataSets/DroneVehicle_adjust/Annotation_cleaned/change_list_from_original/change_list.csv \
  --output annotations/dronevehicle_refined_yolo_obb/annotation_changes.csv
```

To re-check the released YOLO-OBB labels:

```bash
python tools/check_dronevehicle_refined_annotations.py \
  --label-root annotations/dronevehicle_refined_yolo_obb/labels \
  --allow-empty \
  --stats-json annotations/dronevehicle_refined_yolo_obb/label_stats.json
```
