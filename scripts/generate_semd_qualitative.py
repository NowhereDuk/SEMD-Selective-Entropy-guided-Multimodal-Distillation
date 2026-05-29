#!/usr/bin/env python3
"""Generate qualitative figures for SEMD detection, gate, and entropy analysis."""

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402


DATA_ROOT = REPO_ROOT / "datasets" / "DroneVehicle_rgbir_obb"
DEFAULT_OUT = REPO_ROOT / "outputs" / "semd_qualitative"
DEFAULT_SAMPLES = ("01734", "07913", "08619", "08693", "05937", "06364")

FULL_SEMD = REPO_ROOT / (
    "runs/obb/semd_default/weights/best.pt"
)
ABLATION_BASELINE = REPO_ROOT / (
    "runs/obb_ablation/ablation_no_entropy_guided_fusion/weights/best.pt"
)
WO_ENTROPY_WEIGHT = REPO_ROOT / (
    "runs/obb_ablation/ablation_without_student_entropy_weight/weights/best.pt"
)
RGB_TEACHER = REPO_ROOT / "weight/teacher_hgnetv2_obb_rgb/best.pt"
IR_TEACHER = REPO_ROOT / "weight/teacher_hgnetv2_obb_ir/best.pt"
C2FORMER_RESULTS = Path(
    REPO_ROOT / "external_results/C2Former/"
    "test_epoch24_map50_95/c2former_epoch24_test_results.pkl"
)
C2FORMER_ANN_GLOB = str(REPO_ROOT / "external_results/C2Former/annfiles/*_tir.txt")
E2E_MFD_RESULTS = Path(
    REPO_ROOT / "external_results/E2E-MFD/"
    "test_best_epoch17_3gpu_spg4/results.pkl"
)
E2E_MFD_ANN_GLOB = str(REPO_ROOT / "external_results/E2E-MFD/labels/*.txt")

CLASS_NAMES = {
    0: "car",
    1: "freight_car",
    2: "truck",
    3: "bus",
    4: "van",
}
CLASS_COLORS = {
    0: (67, 160, 71),
    1: (251, 140, 0),
    2: (30, 136, 229),
    3: (142, 36, 170),
    4: (211, 47, 47),
}
FUSION_LAYERS = {
    "p3": 10,
    "p4": 13,
    "p5": 16,
}
C2FORMER_CLASS_MAP = {
    0: 0,  # car
    1: 2,  # truck
    2: 1,  # freight_car
    3: 3,  # bus
    4: 4,  # van
}
E2E_MFD_CLASS_MAP = {
    0: 0,  # car
    1: 1,  # freight_car
    2: 2,  # truck
    3: 3,  # bus
    4: 4,  # van
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES), help="Comma-separated sample IDs.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument("--device", default="0", help="Ultralytics device argument, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--full-semd", type=Path, default=FULL_SEMD)
    parser.add_argument("--ablation-baseline", type=Path, default=ABLATION_BASELINE)
    parser.add_argument("--wo-entropy-weight", type=Path, default=WO_ENTROPY_WEIGHT)
    parser.add_argument("--rgb-teacher", type=Path, default=RGB_TEACHER)
    parser.add_argument("--ir-teacher", type=Path, default=IR_TEACHER)
    parser.add_argument("--skip-teacher-preference", action="store_true")
    parser.add_argument("--paper-compact", action="store_true", help="Generate compact paper-ready overview figures.")
    parser.add_argument(
        "--paper-entropy-compact-only",
        action="store_true",
        help="Only rebuild the compact paper entropy overview from existing per-sample entropy images.",
    )
    parser.add_argument(
        "--paper-show-box-labels",
        action="store_true",
        help="Show per-box class/confidence text in --paper-compact figures.",
    )
    parser.add_argument("--paper-panel-height", type=int, default=260, help="Panel height used by --paper-compact.")
    parser.add_argument("--c2former-results", type=Path, default=C2FORMER_RESULTS)
    parser.add_argument("--e2e-mfd-results", type=Path, default=E2E_MFD_RESULTS)
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> None:
    if args.paper_entropy_compact_only:
        required = [
            args.data_root / "images" / "test",
            args.data_root / "images_ir" / "test",
            args.out / "entropy",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
        return

    required = [
        args.data_root / "images" / "test",
        args.data_root / "images_ir" / "test",
        args.data_root / "labels" / "test",
        args.full_semd,
        args.ablation_baseline,
        args.wo_entropy_weight,
    ]
    if not args.skip_teacher_preference:
        required.extend([args.rgb_teacher, args.ir_teacher])
    if args.paper_compact:
        required.extend([args.c2former_results, args.e2e_mfd_results])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def sample_ids(raw: str) -> list[str]:
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not ids:
        raise ValueError("--samples must contain at least one sample id")
    return ids


def mkdirs(out: Path) -> None:
    for name in ("detection", "gate", "teacher_preference", "entropy", "supplement"):
        (out / name).mkdir(parents=True, exist_ok=True)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image


def image_paths(data_root: Path, stem: str) -> tuple[Path, Path, Path]:
    rgb = data_root / "images" / "test" / f"{stem}.jpg"
    ir = data_root / "images_ir" / "test" / f"{stem}.jpg"
    label = data_root / "labels" / "test" / f"{stem}.txt"
    if not rgb.exists() or not ir.exists() or not label.exists():
        raise FileNotFoundError(f"Missing RGB/IR/label for sample {stem}")
    return rgb, ir, label


def parse_gt_obb(label_path: Path, shape: tuple[int, int]) -> list[tuple[int, np.ndarray]]:
    h, w = shape
    boxes: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 9:
            raise ValueError(f"Expected YOLO OBB label with 9 values in {label_path}: {line}")
        cls = int(float(parts[0]))
        coords = np.asarray([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        boxes.append((cls, coords))
    return boxes


def draw_text_box(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.45,
    thickness: int = 1,
) -> None:
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - th - baseline - 3), (x + tw + 4, y + 2), color, -1)
    cv2.putText(image, text, (x + 2, y - baseline), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_polygons(
    image_rgb: np.ndarray,
    boxes: Iterable[tuple[int, np.ndarray, float | None]],
    show_conf: bool,
    show_label: bool = True,
    line_width: int | None = None,
) -> np.ndarray:
    out = image_rgb.copy()
    h, w = out.shape[:2]
    lw = line_width or max(2, round((h + w) / 700))
    for cls, poly, conf in boxes:
        color = CLASS_COLORS.get(int(cls), (255, 255, 255))
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=lw, lineType=cv2.LINE_AA)
        if not show_label:
            continue
        label = CLASS_NAMES.get(int(cls), str(int(cls)))
        if show_conf and conf is not None:
            label = f"{label} {conf:.2f}"
        anchor = tuple(np.clip(pts.reshape(-1, 2).min(axis=0), [0, 16], [w - 1, h - 1]).astype(int))
        draw_text_box(out, label, anchor, color, scale=max(0.4, min(0.65, lw * 0.18)), thickness=1)
    return out


def result_boxes(result) -> list[tuple[int, np.ndarray, float]]:
    if getattr(result, "obb", None) is None or len(result.obb) == 0:
        return []
    obb = result.obb.cpu()
    polys = obb.xyxyxyxy.numpy()
    cls = obb.cls.numpy().astype(int)
    conf = obb.conf.numpy()
    return [(int(c), poly.astype(np.float32), float(s)) for c, poly, s in zip(cls, polys, conf)]


def rotated_box_to_poly(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h, angle = [float(x) for x in box[:5]]
    dx, dy = w * 0.5, h * 0.5
    corners = np.asarray([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return corners @ rot.T + np.asarray([cx, cy], dtype=np.float32)


def sample_id_from_ann(path: str, suffix: str = "") -> str:
    stem = Path(path).stem
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem.split("_")[0]


def load_mmrotate_result_boxes(
    result_path: Path,
    ann_glob: str,
    class_map: dict[int, int],
    score_thr: float,
    ann_suffix: str = "",
) -> dict[str, list[tuple[int, np.ndarray, float]]]:
    ann_files = glob.glob(ann_glob)
    if not ann_files:
        raise FileNotFoundError(f"No annotation files matched: {ann_glob}")
    with result_path.open("rb") as f:
        results = pickle.load(f)
    if len(results) != len(ann_files):
        raise ValueError(f"{result_path} has {len(results)} results but {ann_glob} matched {len(ann_files)} files")

    by_sample: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    for ann_file, per_image in zip(ann_files, results):
        stem = sample_id_from_ann(ann_file, ann_suffix)
        boxes: list[tuple[int, np.ndarray, float]] = []
        for src_cls, arr in enumerate(per_image):
            mapped_cls = class_map[src_cls]
            arr = np.asarray(arr)
            if arr.size == 0:
                continue
            for row in arr:
                score = float(row[5])
                if score < score_thr:
                    continue
                boxes.append((mapped_cls, rotated_box_to_poly(row), score))
        by_sample[stem] = boxes
    return by_sample


def predict_one(model: YOLO, img_path: Path, args: argparse.Namespace):
    outputs = model.predict(
        source=str(img_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        ch=6,
        device=args.device,
        save=False,
        verbose=False,
    )
    if not outputs:
        raise RuntimeError(f"No prediction output for {img_path}")
    if isinstance(outputs[0], (list, tuple)):
        return outputs[0][0]
    return outputs[0]


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    bar_h = 42
    out = np.full((image.shape[0] + bar_h, image.shape[1], 3), 255, dtype=np.uint8)
    out[bar_h:] = image
    cv2.putText(out, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_row(panels: list[tuple[str, np.ndarray]], height: int = 360, gap: int = 8) -> np.ndarray:
    titled = [add_title(resize_to_height(img, height), title) for title, img in panels]
    max_h = max(img.shape[0] for img in titled)
    padded = []
    for img in titled:
        if img.shape[0] < max_h:
            pad = np.full((max_h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
    separator = np.full((max_h, gap, 3), 255, dtype=np.uint8)
    row = padded[0]
    for panel in padded[1:]:
        row = np.hstack([row, separator, panel])
    return row


def stack_rows(rows: list[np.ndarray], gap: int = 12) -> np.ndarray:
    max_w = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.hstack([row, pad])
        padded_rows.append(row)
    separator = np.full((gap, max_w, 3), 255, dtype=np.uint8)
    out = padded_rows[0]
    for row in padded_rows[1:]:
        out = np.vstack([out, separator, row])
    return out


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_class_color_legend(path: Path) -> None:
    swatch_w, swatch_h = 54, 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = []
    for cls in sorted(CLASS_NAMES):
        label = CLASS_NAMES[cls]
        color = CLASS_COLORS[cls]
        row = np.full((40, 250, 3), 255, dtype=np.uint8)
        cv2.rectangle(row, (16, 12), (16 + swatch_w, 12 + swatch_h), color, -1)
        cv2.rectangle(row, (16, 12), (16 + swatch_w, 12 + swatch_h), (30, 30, 30), 1)
        cv2.putText(row, label, (88, 29), font, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(row)
    legend = np.vstack(rows)
    save_rgb(path, legend)


def normalize_map(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def apply_colormap(x: np.ndarray, cmap: str = "magma") -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    rgb = plt.get_cmap(cmap)(x)[..., :3]
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_rgb = cv2.resize(heat_rgb, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(base_rgb.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha, 0, 255).astype(
        np.uint8
    )


def preference_overlay(base_rgb: np.ndarray, rgb_gate: np.ndarray, ir_gate: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    pref = np.zeros_like(base_rgb, dtype=np.float32)
    pref[..., 2] = np.clip(rgb_gate, 0.0, 1.0) * 255.0
    pref[..., 0] = np.clip(ir_gate, 0.0, 1.0) * 255.0
    return np.clip(base_rgb.astype(np.float32) * (1.0 - alpha) + pref * alpha, 0, 255).astype(np.uint8)


def cache_tensor_to_map(tensor: torch.Tensor, channel: int | None, out_hw: tuple[int, int]) -> np.ndarray:
    data = tensor.detach().float().cpu()
    if data.ndim == 4:
        data = data[0]
    if channel is not None:
        data = data[channel]
    elif data.ndim == 3:
        data = data.mean(0)
    arr = data.numpy().astype(np.float32)
    return cv2.resize(arr, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)


def load_teacher(model_path: Path, device: str) -> YOLO:
    return YOLO(str(model_path))


def teacher_confidence(model: YOLO, image_rgb: np.ndarray, device: str, imgsz: int) -> np.ndarray:
    """Return a class-agnostic confidence map from teacher head feature maps."""
    # This uses a minimal image path independent forward. It follows the predictor preprocessing
    # enough for visualization; the confidence map is qualitative rather than metric-bearing.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h0, w0 = image_bgr.shape[:2]
    scale = imgsz / max(h0, w0)
    resized = cv2.resize(image_bgr, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    x = canvas[..., ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x)[None]
    device_obj = torch.device("cpu" if str(device).lower() == "cpu" else f"cuda:{str(device).split(',')[0]}")
    model.model.to(device_obj).eval()
    tensor = torch.from_numpy(x).to(device_obj).float() / 255.0
    with torch.no_grad():
        preds = model.model(tensor)
    raw = preds[1] if isinstance(preds, tuple) and len(preds) == 2 else preds
    feats = raw[0] if isinstance(raw, tuple) else raw
    if not isinstance(feats, (list, tuple)):
        raise RuntimeError("Teacher forward did not return feature maps")
    head = model.model.model[-1]
    conf = None
    # P3 is the first detection feature map.
    feat = feats[0]
    _, score_map = feat.split((head.reg_max * 4, head.nc), dim=1)
    conf = score_map.sigmoid().amax(dim=1, keepdim=True)
    conf_np = conf[0, 0].detach().float().cpu().numpy()
    conf_np = cv2.resize(conf_np, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    conf_np = conf_np[: resized.shape[0], : resized.shape[1]]
    return cv2.resize(conf_np, (w0, h0), interpolation=cv2.INTER_LINEAR)


def generate_detection_figures(
    samples: list[str],
    models: dict[str, YOLO],
    args: argparse.Namespace,
    external_results: dict[str, dict[str, list[tuple[int, np.ndarray, float]]]] | None = None,
) -> dict[str, np.ndarray]:
    overview_rows = []
    semd_panels: dict[str, np.ndarray] = {}
    panel_height = args.paper_panel_height if args.paper_compact else 320
    show_box_labels = (not args.paper_compact) or args.paper_show_box_labels
    for stem in samples:
        rgb_path, ir_path, label_path = image_paths(args.data_root, stem)
        rgb = read_rgb(rgb_path)
        ir = read_rgb(ir_path)
        gt = draw_polygons(
            rgb,
            [(cls, poly, None) for cls, poly in parse_gt_obb(label_path, rgb.shape[:2])],
            False,
            show_label=show_box_labels,
        )

        pred_panels = {}
        if args.paper_compact:
            if external_results is None:
                raise RuntimeError("--paper-compact requires external comparison results")
            for key in ("C2Former", "E2E-MFD"):
                boxes = external_results[key].get(stem)
                if boxes is None:
                    raise KeyError(f"Sample {stem} was not found in {key} external results")
                pred_panels[key] = draw_polygons(rgb, boxes, True, show_label=show_box_labels)
            result = predict_one(models["SEMD"], rgb_path, args)
            pred_panels["SEMD"] = draw_polygons(rgb, result_boxes(result), True, show_label=show_box_labels)
            semd_panels[stem] = pred_panels["SEMD"]
            panels = [
                ("RGB", rgb),
                ("IR", ir),
                ("GT", gt),
                ("C2Former", pred_panels["C2Former"]),
                ("E2E-MFD", pred_panels["E2E-MFD"]),
                ("SEMD", pred_panels["SEMD"]),
            ]
            row = make_row(panels, height=panel_height)
            save_rgb(args.out / "detection" / f"{stem}_comparison_paper.png", row)
        else:
            for key, model in models.items():
                result = predict_one(model, rgb_path, args)
                pred_panels[key] = draw_polygons(rgb, result_boxes(result), True, show_label=show_box_labels)

            semd_panels[stem] = pred_panels["SEMD"]
            panels = [
                ("RGB", rgb),
                ("IR", ir),
                ("GT", gt),
                ("Ablation baseline", pred_panels["Ablation baseline"]),
                ("w/o entropy weight", pred_panels["w/o entropy weight"]),
                ("SEMD", pred_panels["SEMD"]),
            ]
            row = make_row(panels, height=panel_height)
            save_rgb(args.out / "detection" / f"{stem}_comparison.png", row)
        overview_rows.append(row)
    overview = stack_rows(overview_rows)
    if args.paper_compact:
        save_rgb(args.out / "figure_detection_comparison_paper.png", overview)
    else:
        save_rgb(args.out / "figure_detection_comparison.png", overview)
    return semd_panels


def generate_gate_entropy_figures(
    samples: list[str],
    semd_model: YOLO,
    semd_panels: dict[str, np.ndarray],
    args: argparse.Namespace,
    teachers: tuple[YOLO, YOLO] | None,
) -> None:
    gate_rows = []
    entropy_rows = []
    panel_height = args.paper_panel_height if args.paper_compact else 320
    for stem in samples:
        rgb_path, ir_path, _ = image_paths(args.data_root, stem)
        rgb = read_rgb(rgb_path)
        ir = read_rgb(ir_path)

        _ = predict_one(semd_model, rgb_path, args)
        model = semd_model.model
        for scale_name, layer_idx in FUSION_LAYERS.items():
            cache = getattr(model.model[layer_idx], "kd_cache", {}) or {}
            required = {"gate_probs", "gate_entropy"}
            if not required.issubset(cache):
                raise RuntimeError(f"Missing kd_cache keys at layer {layer_idx}: {sorted(cache.keys())}")
            gate_probs = cache["gate_probs"]
            gate_entropy = cache["gate_entropy"]

            rgb_gate = cache_tensor_to_map(gate_probs, 0, rgb.shape[:2])
            ir_gate = cache_tensor_to_map(gate_probs, 1, rgb.shape[:2])
            entropy = cache_tensor_to_map(gate_entropy, None, rgb.shape[:2])
            entropy = np.clip(entropy, 0.0, 1.0)
            weight = np.clip(1.0 + 1.0 * np.maximum(entropy - 0.5, 0.0) - 0.5 * np.maximum(0.5 - entropy, 0.0), 0.5, 1.5)
            weight_norm = (weight - 0.5) / 1.0

            rgb_gate_img = apply_colormap(np.clip(rgb_gate, 0.0, 1.0), "Blues")
            ir_gate_img = apply_colormap(np.clip(ir_gate, 0.0, 1.0), "Reds")
            gate_overlay = preference_overlay(rgb, rgb_gate, ir_gate)
            entropy_img = apply_colormap(entropy, "magma")
            weight_img = apply_colormap(weight_norm, "inferno")
            entropy_overlay = overlay_heatmap(rgb, entropy_img)

            suffix = scale_name
            save_rgb(args.out / "gate" / f"{stem}_{suffix}_rgb_gate.png", rgb_gate_img)
            save_rgb(args.out / "gate" / f"{stem}_{suffix}_ir_gate.png", ir_gate_img)
            save_rgb(args.out / "gate" / f"{stem}_{suffix}_gate_overlay.png", gate_overlay)
            save_rgb(args.out / "entropy" / f"{stem}_{suffix}_gate_entropy.png", entropy_img)
            save_rgb(args.out / "entropy" / f"{stem}_{suffix}_entropy_weight.png", weight_img)
            save_rgb(args.out / "entropy" / f"{stem}_{suffix}_entropy_overlay.png", entropy_overlay)

            if scale_name != "p3":
                save_rgb(args.out / "supplement" / f"{stem}_{suffix}_gate_entropy_overlay.png", entropy_overlay)

        teacher_panel = None
        if teachers is not None:
            rgb_conf = teacher_confidence(teachers[0], rgb, args.device, args.imgsz)
            ir_conf = teacher_confidence(teachers[1], ir, args.device, args.imgsz)
            teacher_pref = rgb_conf / np.maximum(rgb_conf + ir_conf, 1e-6)
            teacher_pref_img = apply_colormap(teacher_pref, "coolwarm_r")
            teacher_panel = overlay_heatmap(rgb, teacher_pref_img)
            save_rgb(args.out / "teacher_preference" / f"{stem}_p3_teacher_pref.png", teacher_panel)

        # Re-read P3 products for overview.
        p3_rgb_gate = read_rgb(args.out / "gate" / f"{stem}_p3_rgb_gate.png")
        p3_ir_gate = read_rgb(args.out / "gate" / f"{stem}_p3_ir_gate.png")
        p3_gate_overlay = read_rgb(args.out / "gate" / f"{stem}_p3_gate_overlay.png")
        p3_entropy = read_rgb(args.out / "entropy" / f"{stem}_p3_gate_entropy.png")
        p3_weight = read_rgb(args.out / "entropy" / f"{stem}_p3_entropy_weight.png")

        if args.paper_compact:
            gate_panels = [
                ("RGB", rgb),
                ("IR", ir),
                ("Student preference", p3_gate_overlay),
            ]
            if teacher_panel is not None:
                gate_panels.append(("Teacher preference", teacher_panel))
        else:
            gate_panels = [
                ("RGB", rgb),
                ("IR", ir),
                ("RGB gate", p3_rgb_gate),
                ("IR gate", p3_ir_gate),
                ("Student preference", p3_gate_overlay),
            ]
            if teacher_panel is not None:
                gate_panels.append(("Teacher preference", teacher_panel))
        gate_rows.append(make_row(gate_panels, height=panel_height))

        entropy_rows.append(
            make_row(
                [
                    ("RGB", rgb),
                    ("IR", ir),
                    ("Student gate entropy", p3_entropy),
                    ("Entropy weight", p3_weight),
                    ("SEMD detection", semd_panels[stem]),
                ],
                height=panel_height,
            )
        )

    if args.paper_compact:
        save_rgb(args.out / "figure_gate_preference_paper.png", stack_rows(gate_rows))
        save_rgb(args.out / "figure_gate_entropy_paper.png", stack_rows(entropy_rows))
    else:
        save_rgb(args.out / "figure_gate_preference.png", stack_rows(gate_rows))
        save_rgb(args.out / "figure_gate_entropy.png", stack_rows(entropy_rows))


def generate_entropy_compact_only(samples: list[str], args: argparse.Namespace) -> None:
    figure_path = args.out / "figure_gate_entropy_paper.png"
    backup_path = args.out / "figure_gate_entropy_paper_5col4row_backup.png"
    if figure_path.exists() and not backup_path.exists():
        shutil.copy2(figure_path, backup_path)

    rows = []
    for stem in samples:
        rgb_path = args.data_root / "images" / "test" / f"{stem}.jpg"
        ir_path = args.data_root / "images_ir" / "test" / f"{stem}.jpg"
        entropy_path = args.out / "entropy" / f"{stem}_p3_gate_entropy.png"
        weight_path = args.out / "entropy" / f"{stem}_p3_entropy_weight.png"
        missing = [p for p in (rgb_path, ir_path, entropy_path, weight_path) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing compact entropy assets:\n" + "\n".join(str(p) for p in missing))

        rows.append(
            make_row(
                [
                    ("RGB", read_rgb(rgb_path)),
                    ("IR", read_rgb(ir_path)),
                    ("Student gate entropy", read_rgb(entropy_path)),
                    ("Entropy weight", read_rgb(weight_path)),
                ],
                height=args.paper_panel_height,
            )
        )

    save_rgb(figure_path, stack_rows(rows))
    if backup_path.exists():
        print(f"[SEMD qualitative] backup: {backup_path}")


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    args = parse_args()
    args.out = args.out.resolve()
    mkdirs(args.out)
    ensure_paths(args)
    samples = sample_ids(args.samples)

    print(f"[SEMD qualitative] samples: {', '.join(samples)}")
    print(f"[SEMD qualitative] output: {args.out}")

    if args.paper_entropy_compact_only:
        generate_entropy_compact_only(samples, args)
        print("[SEMD qualitative] done")
        print(f"  entropy:   {args.out / 'figure_gate_entropy_paper.png'}")
        return

    if args.paper_compact:
        models = {"SEMD": YOLO(str(args.full_semd))}
        external_results = {
            "C2Former": load_mmrotate_result_boxes(
                args.c2former_results,
                C2FORMER_ANN_GLOB,
                C2FORMER_CLASS_MAP,
                args.conf,
                ann_suffix="_tir",
            ),
            "E2E-MFD": load_mmrotate_result_boxes(
                args.e2e_mfd_results,
                E2E_MFD_ANN_GLOB,
                E2E_MFD_CLASS_MAP,
                args.conf,
            ),
        }
    else:
        models = {
            "Ablation baseline": YOLO(str(args.ablation_baseline)),
            "w/o entropy weight": YOLO(str(args.wo_entropy_weight)),
            "SEMD": YOLO(str(args.full_semd)),
        }
        external_results = None
    semd_panels = generate_detection_figures(samples, models, args, external_results)
    if args.paper_compact and not args.paper_show_box_labels:
        save_class_color_legend(args.out / "class_color_legend.png")

    teachers = None
    if not args.skip_teacher_preference:
        teachers = (load_teacher(args.rgb_teacher, args.device), load_teacher(args.ir_teacher, args.device))
    generate_gate_entropy_figures(samples, models["SEMD"], semd_panels, args, teachers)

    print("[SEMD qualitative] done")
    if args.paper_compact:
        print(f"  detection: {args.out / 'figure_detection_comparison_paper.png'}")
        print(f"  gate:      {args.out / 'figure_gate_preference_paper.png'}")
        print(f"  entropy:   {args.out / 'figure_gate_entropy_paper.png'}")
    else:
        print(f"  detection: {args.out / 'figure_detection_comparison.png'}")
        print(f"  gate:      {args.out / 'figure_gate_preference.png'}")
        print(f"  entropy:   {args.out / 'figure_gate_entropy.png'}")


if __name__ == "__main__":
    main()
