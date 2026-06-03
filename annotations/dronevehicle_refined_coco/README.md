# SEMD Refined DroneVehicle COCO Annotations

This directory contains modality-specific cleaned COCO JSON annotations from:

```text
/home/disk1/DataSets/DroneVehicle_adjust/Annotation_cleaned
```

The JSON files are provided for users who need COCO-style annotations or want to
audit the modality-specific cleaned annotations. SEMD student training uses the
shared YOLO-OBB labels in:

```text
annotations/dronevehicle_refined_yolo_obb/labels/
```

## Files

```text
DV_train_rgb_basename.json
DV_train_ir_basename.json
DV_val_rgb_basename.json
DV_val_ir_basename.json
DV_test_rgb_basename.json
DV_test_ir_basename.json
cleaning_summary.json
```

The `*_basename.json` files use image basenames so they can be matched to local
DroneVehicle images after users arrange the dataset in their own directory.

## Notes

- Original DroneVehicle images are not included.
- Use the official DroneVehicle page to obtain the images.
- Cite DroneVehicle when using these refined annotations.
