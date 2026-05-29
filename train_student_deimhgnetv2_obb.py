#!/usr/bin/env python3
"""Train the dual-stream HGNetv2 OBB student with switchable OBB teacher backbones."""

import argparse
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel, attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "model_yaml_obb" / "yolov8-EntropyOffsetGate-deimhgnetv2-b0-obb.yaml"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "DroneVehicle_obb_student_external.yaml"
DEFAULT_SAVE_DIR = REPO_ROOT / "runs" / "obb" / "dronevehicle_student_dist_deimhgnetv2_obb"
DEFAULT_TEACHER_CHECKPOINTS = {
    "hgnetv2": {
        "rgb": REPO_ROOT / "runs" / "obb" / "teacher_hgnetv2_obb_rgb" / "weights" / "best.pt",
        "ir": REPO_ROOT / "runs" / "obb" / "teacher_hgnetv2_obb_ir" / "weights" / "best.pt",
    },
    "yolov8": {
        "rgb": REPO_ROOT / "teacherTraining" / "runs" / "obb" / "train6" / "weights" / "best.pt",
        "ir": REPO_ROOT / "teacherTraining" / "runs" / "obb" / "train5" / "weights" / "best.pt",
    },
}
DEFAULT_STUDENT_TAPS = {
    "layers_rgb": [6, 8, 11, 14],
    "layers_ir": [7, 9, 12, 15],
    "layers_fusion": [10, 13, 16],
}
ADDFUSION_STUDENT_TAPS = {
    "layers_rgb": [5, 7, 10, 13],
    "layers_ir": [6, 8, 11, 14],
    "layers_fusion": [9, 12, 15],
}
P2_STUDENT_TAPS = {
    "layers_rgb": [6, 9, 12, 15],
    "layers_ir": [7, 10, 13, 16],
    "layers_fusion": [8, 11, 14, 17],
}
TEACHER_TAPS = {
    "yolov8": [4, 6, 8],
    "hgnetv2": [1, 2, 3, 4],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the dual-stream HGNetv2 OBB student with multi-teacher distillation."
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a saved checkpoint path such as runs/.../weights/last.pt.")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Optional checkpoint used for shape-matched initialization with the requested YAML.")
    parser.add_argument("--teacher-arch", choices=("yolov8", "hgnetv2"), default="hgnetv2")
    parser.add_argument("--teacher-rgb", type=str, default=None, help="Optional RGB teacher checkpoint override.")
    parser.add_argument("--teacher-ir", type=str, default=None, help="Optional IR teacher checkpoint override.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=132)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--optimizer",
        choices=("SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSProp", "auto"),
        default="auto",
    )
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--loss-type", choices=("CWD", "PKD"), default="CWD")
    parser.add_argument("--distill-weight", type=float, default=0.8)
    parser.add_argument("--distill-disable-all", action=argparse.BooleanOptionalAction, default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument("--distill-cross-attention", action=argparse.BooleanOptionalAction, default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--distill-normal-distillation", action=argparse.BooleanOptionalAction, default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--distill-only-normal", action=argparse.BooleanOptionalAction, default=False,
                        help="Strictly keep only same-modality branch KD (RGB<-RGB, IR<-IR).")
    parser.add_argument("--distill-disable-gate-kd", action=argparse.BooleanOptionalAction, default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument("--distill-disable-pseudo-fusion-kd", action=argparse.BooleanOptionalAction, default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument("--distill-head-kd-policy", type=str, default="full",
                        choices=("full", "cls_only", "geom_only", "off"),
                        help="Head KD policy for cls/dfl/angle distillation.")
    parser.add_argument("--distill-epoch-scale-mode", type=str, default="legacy_cosine",
                        choices=("legacy_cosine", "cosine", "linear", "none"),
                        help="Epoch-level teacher influence schedule. legacy_cosine preserves the old hard-coded schedule.")
    parser.add_argument("--distill-epoch-scale-start", type=float, default=1.0,
                        help="Start scale for cosine/linear epoch-level KD decay.")
    parser.add_argument("--distill-epoch-scale-end", type=float, default=0.1,
                        help="End scale for cosine/linear epoch-level KD decay.")
    parser.add_argument("--distill-epoch-scale-decay-start", type=int, default=0,
                        help="Epoch index where cosine/linear KD decay starts.")
    parser.add_argument("--distill-epoch-scale-decay-end", type=int, default=-1,
                        help="Epoch index where cosine/linear KD decay ends; -1 means total training epochs.")
    parser.add_argument("--distill-zero-after-epoch", type=int, default=-1,
                        help="Set KD epoch scale to zero at and after this epoch; -1 disables it.")
    parser.add_argument("--distill-schedule-enable", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable epoch-aware KD schedule transitions.")
    parser.add_argument("--distill-mid-kd-start-epoch", type=int, default=-1,
                        help="Absolute epoch index that activates the mid-stage KD policy; -1 disables it.")
    parser.add_argument("--distill-mid-kd-policy", type=str, default="none",
                        choices=("none", "normal_only_gate_off"),
                        help="Mid-stage KD policy applied after --distill-mid-kd-start-epoch.")
    parser.add_argument("--distill-gate-kd-mode", type=str, default="legacy",
                        choices=("legacy", "normalized"),
                        help="Gate KD reduction mode.")
    parser.add_argument("--distill-gate-kd-weight", type=float, default=1.0,
                        help="Additional scale applied to gate KD.")
    parser.add_argument("--distill-gate-kd-temperature", type=float, default=1.0,
                        help="Temperature used by gate KD.")
    parser.add_argument("--distill-gate-kd-mask-mode", type=str, default="none",
                        choices=("none", "conf_binary", "conf_soft", "conf_binary_soft"),
                        help="Reliability mask mode for normalized gate KD.")
    parser.add_argument("--distill-gate-kd-conf-thr", type=float, default=0.25,
                        help="Confidence threshold used by masked gate KD modes.")
    parser.add_argument("--distill-student-entropy-weight-enable", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable student-entropy-based KD weighting.")
    parser.add_argument("--distill-student-entropy-weight-mode", type=str, default="fixed",
                        choices=("fixed", "learnable"),
                        help="How student entropy is converted into a KD weight.")
    parser.add_argument("--distill-student-entropy-weight-formula", type=str, default="fixed_boost",
                        choices=("fixed_boost", "centered", "linear_map", "asym_centered"),
                        help="Formula used by fixed student entropy weighting.")
    parser.add_argument("--distill-student-entropy-weight-min", type=float, default=0.5,
                        help="Minimum bounded KD weight from student entropy weighting.")
    parser.add_argument("--distill-student-entropy-weight-max", type=float, default=1.5,
                        help="Maximum bounded KD weight from student entropy weighting.")
    parser.add_argument("--distill-student-entropy-weight-beta", type=float, default=0.5,
                        help="Slope used by fixed student entropy weighting.")
    parser.add_argument("--distill-student-entropy-weight-beta-pos", type=float, default=1.0,
                        help="Positive-side slope used by asym_centered student entropy weighting.")
    parser.add_argument("--distill-student-entropy-weight-beta-neg", type=float, default=0.5,
                        help="Negative-side slope used by asym_centered student entropy weighting.")
    parser.add_argument("--distill-student-entropy-weight-tau", type=float, default=0.5,
                        help="Center point used by centered student entropy weighting.")
    parser.add_argument("--distill-student-entropy-weight-normalize-mean",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Normalize entropy-derived KD weights by their valid-region mean.")
    parser.add_argument("--distill-student-entropy-weight-detach", action=argparse.BooleanOptionalAction, default=True,
                        help="Detach student entropy before converting it into a KD weight.")
    parser.add_argument("--distill-student-entropy-weight-target", type=str, default="gate",
                        choices=("gate", "head_cls", "gate_head_cls"),
                        help="Which KD branch should use student-entropy weighting.")
    parser.add_argument("--distill-student-entropy-weight-reg", type=float, default=0.0,
                        help="Optional regularization strength for entropy-derived KD weights.")
    parser.add_argument("--distill-late-kd-start-epoch", type=int, default=-1,
                        help="Absolute epoch index that activates late KD scheduling; -1 disables it.")
    parser.add_argument("--distill-late-kd-policy", type=str, default="none",
                        choices=("none", "gate_off", "normal_gate_off", "strict_only_normal", "strict_head_kd_off"),
                        help="Late KD policy applied after --distill-late-kd-start-epoch.")
    parser.add_argument("--distill-stage2-cls-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--distill-resume-ckpt", type=str, default=None)
    parser.add_argument("--freeze-backbone-fusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-neck-head-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage2-lr-mult", type=float, default=0.1)
    parser.add_argument("--stage2-epochs", type=int, default=25)
    parser.add_argument("--stage2-disable-early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distill-cls-kd-weight", type=float, default=0.05)
    parser.add_argument("--distill-kd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-conf-thr", type=float, default=0.45)
    parser.add_argument("--teacher-entropy-thr", type=float, default=0.35)
    parser.add_argument("--teacher-jsd-thr", type=float, default=0.10)
    parser.add_argument("--audit-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--audit-disable-early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-ckpt-interval", type=int, default=5)
    parser.add_argument("--audit-probe-epochs", type=int, default=10)
    parser.add_argument("--audit-probe-fractions", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument("--audit-probes", type=str, default="continue,kd_off,normal_only,cross_off,lr_x0.3")
    parser.add_argument("--audit-log-every-n", type=int, default=200)
    parser.add_argument("--audit-output-name", type=str, default="headroom_audit")
    parser.add_argument("--audit-eval-split", type=str, default="val", choices=("val", "test"))
    parser.add_argument("--online", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_yaml_dict(path_str):
    yaml_path = resolve_path(path_str)
    if not yaml_path.exists():
        raise FileNotFoundError(f"yaml not found: {yaml_path}")
    return yaml_path, yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}


def resolve_dataset_root(data_yaml_path):
    data_yaml_path, data = load_yaml_dict(data_yaml_path)
    raw_root = str(data.get("path", "")).strip()
    if not raw_root:
        raise ValueError(f"dataset yaml missing 'path': {data_yaml_path}")
    dataset_root = Path(raw_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()
    return data_yaml_path, data, dataset_root


def first_non_empty_label(label_dir):
    for label_path in sorted(label_dir.glob("*.txt")):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return label_path, line.strip()
    return None, None


def validate_obb_dataset(data_yaml_path):
    data_yaml_path, data, dataset_root = resolve_dataset_root(data_yaml_path)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    checked_split = False
    for split in ("train", "val"):
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            continue
        checked_split = True
        label_path, sample_line = first_non_empty_label(label_dir)
        if sample_line is None:
            continue
        if len(sample_line.split()) < 9:
            raise ValueError(
                f"dataset is not OBB-ready: {label_path} has fewer than 9 columns "
                f"(expected class + 8 coordinates)."
            )
    if not checked_split:
        raise FileNotFoundError(f"no labels/train or labels/val directories found under {dataset_root}")
    return data_yaml_path, data, dataset_root


def resolve_teacher_paths(teacher_arch, teacher_rgb=None, teacher_ir=None):
    defaults = DEFAULT_TEACHER_CHECKPOINTS[teacher_arch]
    rgb_path = resolve_path(teacher_rgb) if teacher_rgb else defaults["rgb"]
    ir_path = resolve_path(teacher_ir) if teacher_ir else defaults["ir"]
    return rgb_path, ir_path


def load_teacher_model(weight_path):
    weight_path = resolve_path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {weight_path}")
    model, _ = attempt_load_one_weight(str(weight_path), device="cpu")
    normalize_hook_dicts(model)
    return model, weight_path


def _unwrap_model(model):
    return de_parallel(model)


def normalize_hook_dicts(model):
    """Convert deserialized hook dicts back to weakref-able OrderedDicts."""
    hook_attrs = (
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    for module in _unwrap_model(model).modules():
        for attr in hook_attrs:
            value = getattr(module, attr, None)
            if isinstance(value, dict) and not isinstance(value, OrderedDict):
                setattr(module, attr, OrderedDict(value))


def _get_model_device(model):
    try:
        return next(_unwrap_model(model).parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _capture_feature_shapes(model, layer_ids, input_channels, imgsz):
    base = _unwrap_model(model)
    normalize_hook_dicts(base)
    if not hasattr(base, "model"):
        raise AttributeError("Target model has no top-level model list for feature taps.")

    total_layers = len(base.model)
    outputs = [None] * len(layer_ids)
    handles = []

    for slot, idx in enumerate(layer_ids):
        if idx < 0 or idx >= total_layers:
            raise IndexError(f"layer id {idx} out of range [0, {total_layers - 1}] for model with {total_layers} layers")

        def make_hook(position):
            def hook(_m, _i, output):
                outputs[position] = output

            return hook

        handles.append(base.model[idx].register_forward_hook(make_hook(slot)))

    was_training = base.training
    base.eval()
    x = torch.randn(1, input_channels, imgsz, imgsz, device=_get_model_device(base))
    try:
        with torch.no_grad():
            base(x)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            base.train()

    missing = [layer_ids[i] for i, output in enumerate(outputs) if output is None]
    if missing:
        raise RuntimeError(f"failed to capture features for layers: {missing}")
    return [tuple(output.shape) for output in outputs]


def validate_layer_types(model, layer_ids, expected_type, side):
    base = _unwrap_model(model)
    layer_types = [type(layer).__name__ for layer in base.model]
    expected_types = (expected_type,) if isinstance(expected_type, str) else tuple(expected_type)
    mismatches = [idx for idx in layer_ids if layer_types[idx] not in expected_types]
    if mismatches:
        expected_label = " or ".join(expected_types)
        raise RuntimeError(
            f"{side} layer taps {mismatches} are not {expected_label}. "
            f"Observed types: {[layer_types[idx] for idx in mismatches]}"
        )


def validate_layer_indices(model, layer_ids, side):
    base = _unwrap_model(model)
    total_layers = len(base.model)
    invalid = [idx for idx in layer_ids if idx < 0 or idx >= total_layers]
    if invalid:
        raise RuntimeError(f"{side} layer taps {invalid} out of range for model with {total_layers} layers")


def build_student_probe_model(model_path, nc):
    model_path = resolve_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"student model yaml not found: {model_path}")
    return OBBModel(str(model_path), ch=6, nc=nc, verbose=False), model_path


def resolve_student_taps(model_path):
    model_name = resolve_path(model_path).name.lower()
    if "addfusion" in model_name:
        return ADDFUSION_STUDENT_TAPS
    if "-p2-obb" in model_name or "-p2p4-obb" in model_name:
        return P2_STUDENT_TAPS
    return DEFAULT_STUDENT_TAPS


def resolve_student_distill_config(model_path, nc, imgsz=640):
    student_model, model_path = build_student_probe_model(model_path, nc)
    student_taps = resolve_student_taps(model_path)
    rgb_layers = student_taps["layers_rgb"]
    ir_layers = student_taps["layers_ir"]
    fusion_layers = student_taps["layers_fusion"]
    validate_layer_types(student_model, rgb_layers + ir_layers, "DEIMHGStage", "student")
    fusion_type = "Add" if "addfusion" in model_path.name.lower() else (
        "Add",
        "EntropyOffsetGateFusion",
        "EntropyOffsetGateFusionLite",
        "EntropyOffsetGateFusionEfficient",
    )
    validate_layer_types(student_model, fusion_layers, fusion_type, "student fusion")
    rgb_shapes = _capture_feature_shapes(student_model, rgb_layers, input_channels=6, imgsz=imgsz)
    ir_shapes = _capture_feature_shapes(student_model, ir_layers, input_channels=6, imgsz=imgsz)
    fusion_shapes = _capture_feature_shapes(student_model, fusion_layers, input_channels=6, imgsz=imgsz)
    return {
        "model_path": model_path,
        "layers_rgb": rgb_layers,
        "layers_ir": ir_layers,
        "layers_fusion": fusion_layers,
        "channels_rgb": [shape[1] for shape in rgb_shapes],
        "channels_ir": [shape[1] for shape in ir_shapes],
        "channels_fusion": [shape[1] for shape in fusion_shapes],
        "shapes_rgb": rgb_shapes,
        "shapes_ir": ir_shapes,
        "shapes_fusion": fusion_shapes,
    }


def resolve_teacher_distill_config(model, teacher_arch, imgsz=640):
    layer_ids = TEACHER_TAPS[teacher_arch]
    validate_layer_indices(model, layer_ids, "teacher")
    if teacher_arch == "hgnetv2":
        validate_layer_types(model, layer_ids, "DEIMHGStage", "teacher")
    shapes = _capture_feature_shapes(model, layer_ids, input_channels=3, imgsz=imgsz)
    return {
        "layers": layer_ids,
        "channels": [shape[1] for shape in shapes],
        "shapes": shapes,
    }


def main():
    args = parse_args()
    data_yaml_path, data_cfg, dataset_root = validate_obb_dataset(args.data)
    save_dir = resolve_path(args.save_dir)
    student_cfg = resolve_student_distill_config(args.model, nc=int(data_cfg["nc"]), imgsz=args.imgsz)
    stage2_resume_ckpt = resolve_path(args.distill_resume_ckpt) if args.distill_resume_ckpt else None
    distill_disabled = bool(args.distill_disable_all)

    teacher_rgb_model = teacher_ir_model = None
    teacher_rgb_path = teacher_ir_path = None
    teacher_cfg_rgb = {"layers": [], "channels": [], "shapes": []}
    teacher_cfg_ir = {"layers": [], "channels": [], "shapes": []}
    if not distill_disabled:
        teacher_rgb_path, teacher_ir_path = resolve_teacher_paths(args.teacher_arch, args.teacher_rgb, args.teacher_ir)
        teacher_rgb_model, teacher_rgb_path = load_teacher_model(teacher_rgb_path)
        teacher_ir_model, teacher_ir_path = load_teacher_model(teacher_ir_path)
        teacher_cfg_rgb = resolve_teacher_distill_config(teacher_rgb_model, args.teacher_arch, imgsz=args.imgsz)
        teacher_cfg_ir = resolve_teacher_distill_config(teacher_ir_model, args.teacher_arch, imgsz=args.imgsz)

    effective_epochs = args.epochs
    effective_lr0 = args.lr0
    effective_patience = None
    effective_resume = str(resolve_path(args.resume)) if args.resume else False
    effective_save_period = None
    if args.distill_stage2_cls_only:
        if stage2_resume_ckpt is None:
            raise ValueError("--distill-stage2-cls-only requires --distill-resume-ckpt.")
        if not stage2_resume_ckpt.exists():
            raise FileNotFoundError(f"stage2 resume checkpoint not found: {stage2_resume_ckpt}")
        effective_epochs = args.stage2_epochs
        effective_lr0 = args.lr0 * args.stage2_lr_mult
        if not args.resume:
            effective_resume = False
        if args.stage2_disable_early_stop:
            effective_patience = max(args.stage2_epochs + 1, args.stage2_epochs * 2)
    elif args.patience is not None:
        effective_patience = args.patience
    if args.audit_mode:
        effective_save_period = max(int(args.audit_ckpt_interval), 1)
        if args.audit_disable_early_stop:
            effective_patience = max(int(effective_patience or 0), effective_epochs + 1, effective_epochs * 2)

    print(f"[info] student model: {student_cfg['model_path']}")
    print(f"[info] data: {data_yaml_path}")
    print(f"[info] dataset root: {dataset_root}")
    print(f"[info] teacher arch: {args.teacher_arch}")
    print(f"[info] teacher rgb: {teacher_rgb_path}")
    print(f"[info] teacher ir: {teacher_ir_path}")
    print(f"[info] student rgb taps: {student_cfg['layers_rgb']} -> {student_cfg['channels_rgb']} {student_cfg['shapes_rgb']}")
    print(f"[info] student ir taps: {student_cfg['layers_ir']} -> {student_cfg['channels_ir']} {student_cfg['shapes_ir']}")
    print(
        f"[info] student fusion taps: {student_cfg['layers_fusion']} -> "
        f"{student_cfg['channels_fusion']} {student_cfg['shapes_fusion']}"
    )
    print(f"[info] teacher rgb taps: {teacher_cfg_rgb['layers']} -> {teacher_cfg_rgb['channels']} {teacher_cfg_rgb['shapes']}")
    print(f"[info] teacher ir taps: {teacher_cfg_ir['layers']} -> {teacher_cfg_ir['channels']} {teacher_cfg_ir['shapes']}")
    print(f"[info] save dir: {save_dir}")
    print(
        "[info] distill epoch scale: "
        f"mode={args.distill_epoch_scale_mode}, "
        f"start={args.distill_epoch_scale_start}, "
        f"end={args.distill_epoch_scale_end}, "
        f"decay_start={args.distill_epoch_scale_decay_start}, "
        f"decay_end={args.distill_epoch_scale_decay_end}, "
        f"zero_after={args.distill_zero_after_epoch}"
    )
    print(
        "[info] student entropy weighting: "
        f"enable={args.distill_student_entropy_weight_enable}, "
        f"mode={args.distill_student_entropy_weight_mode}, "
        f"formula={args.distill_student_entropy_weight_formula}, "
        f"target={args.distill_student_entropy_weight_target}, "
        f"min={args.distill_student_entropy_weight_min}, "
        f"max={args.distill_student_entropy_weight_max}, "
        f"beta={args.distill_student_entropy_weight_beta}, "
        f"beta_pos={args.distill_student_entropy_weight_beta_pos}, "
        f"beta_neg={args.distill_student_entropy_weight_beta_neg}, "
        f"tau={args.distill_student_entropy_weight_tau}, "
        f"normalize_mean={args.distill_student_entropy_weight_normalize_mean}, "
        f"detach={args.distill_student_entropy_weight_detach}, "
        f"reg={args.distill_student_entropy_weight_reg}"
    )
    if args.distill_stage2_cls_only:
        print(f"[info] stage2 resume ckpt: {stage2_resume_ckpt}")
        print(f"[info] stage2 epochs: {effective_epochs}")
        print(f"[info] stage2 lr0: {effective_lr0}")
    if args.audit_mode:
        print(f"[info] audit mode: {args.audit_mode}")
        print(f"[info] audit ckpt interval: {args.audit_ckpt_interval}")
        print(f"[info] audit probe epochs: {args.audit_probe_epochs}")
        print(f"[info] audit probes: {args.audit_probes}")

    overrides = dict(
        model=str(student_cfg["model_path"]),
        data=str(data_yaml_path),
        task="obb",
        resume=effective_resume,
        pretrained=str(resolve_path(args.pretrained)) if args.pretrained else False,
        amp=args.amp,
        imgsz=args.imgsz,
        epochs=effective_epochs,
        batch=args.batch,
        device=args.device,
        optimizer=args.optimizer,
        lr0=effective_lr0,
        online=args.online,
        augment=args.augment,
        workers=args.workers,
        rect=args.rect,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=args.exist_ok,
        ch=6,
        distill_disable_all=distill_disabled,
        distill_cross_attention=bool(args.distill_cross_attention) and not args.distill_stage2_cls_only,
        distill_normal_distillation=bool(args.distill_normal_distillation) and not args.distill_stage2_cls_only,
        distill_only_normal=bool(args.distill_only_normal) and not args.distill_stage2_cls_only,
        distill_disable_gate_kd=bool(args.distill_disable_gate_kd),
        distill_disable_pseudo_fusion_kd=bool(args.distill_disable_pseudo_fusion_kd),
        distill_head_kd_policy=args.distill_head_kd_policy,
        distill_epoch_scale_mode=args.distill_epoch_scale_mode,
        distill_epoch_scale_start=args.distill_epoch_scale_start,
        distill_epoch_scale_end=args.distill_epoch_scale_end,
        distill_epoch_scale_decay_start=args.distill_epoch_scale_decay_start,
        distill_epoch_scale_decay_end=args.distill_epoch_scale_decay_end,
        distill_zero_after_epoch=args.distill_zero_after_epoch,
        distill_schedule_enable=bool(args.distill_schedule_enable),
        distill_mid_kd_start_epoch=args.distill_mid_kd_start_epoch,
        distill_mid_kd_policy=args.distill_mid_kd_policy,
        distill_gate_kd_mode=args.distill_gate_kd_mode,
        distill_gate_kd_weight=args.distill_gate_kd_weight,
        distill_gate_kd_temperature=args.distill_gate_kd_temperature,
        distill_gate_kd_mask_mode=args.distill_gate_kd_mask_mode,
        distill_gate_kd_conf_thr=args.distill_gate_kd_conf_thr,
        distill_student_entropy_weight_enable=bool(args.distill_student_entropy_weight_enable),
        distill_student_entropy_weight_mode=args.distill_student_entropy_weight_mode,
        distill_student_entropy_weight_formula=args.distill_student_entropy_weight_formula,
        distill_student_entropy_weight_min=args.distill_student_entropy_weight_min,
        distill_student_entropy_weight_max=args.distill_student_entropy_weight_max,
        distill_student_entropy_weight_beta=args.distill_student_entropy_weight_beta,
        distill_student_entropy_weight_beta_pos=args.distill_student_entropy_weight_beta_pos,
        distill_student_entropy_weight_beta_neg=args.distill_student_entropy_weight_beta_neg,
        distill_student_entropy_weight_tau=args.distill_student_entropy_weight_tau,
        distill_student_entropy_weight_normalize_mean=bool(args.distill_student_entropy_weight_normalize_mean),
        distill_student_entropy_weight_detach=bool(args.distill_student_entropy_weight_detach),
        distill_student_entropy_weight_target=args.distill_student_entropy_weight_target,
        distill_student_entropy_weight_reg=args.distill_student_entropy_weight_reg,
        distill_late_kd_start_epoch=args.distill_late_kd_start_epoch,
        distill_late_kd_policy=args.distill_late_kd_policy,
        distill_stage2_cls_only=args.distill_stage2_cls_only,
        distill_resume_ckpt=str(stage2_resume_ckpt) if stage2_resume_ckpt is not None else None,
        freeze_backbone_fusion=args.freeze_backbone_fusion,
        train_neck_head_only=args.train_neck_head_only,
        stage2_lr_mult=args.stage2_lr_mult,
        stage2_epochs=args.stage2_epochs,
        stage2_disable_early_stop=args.stage2_disable_early_stop,
        distill_cls_kd_weight=args.distill_cls_kd_weight,
        distill_kd_temperature=args.distill_kd_temperature,
        teacher_conf_thr=args.teacher_conf_thr,
        teacher_entropy_thr=args.teacher_entropy_thr,
        teacher_jsd_thr=args.teacher_jsd_thr,
        audit_mode=args.audit_mode,
        audit_disable_early_stop=args.audit_disable_early_stop,
        audit_ckpt_interval=args.audit_ckpt_interval,
        audit_probe_epochs=args.audit_probe_epochs,
        audit_probe_fractions=args.audit_probe_fractions,
        audit_probes=args.audit_probes,
        audit_log_every_n=args.audit_log_every_n,
        audit_output_name=args.audit_output_name,
        audit_eval_split=args.audit_eval_split,
        split=args.audit_eval_split if (args.audit_mode or os.environ.get("SEMD_AUDIT_PROBE_CHILD")) else "val",
        save_dir=str(save_dir),
    )
    if not distill_disabled:
        overrides.update(
            Distillation="MultiDistillation",
            distill_weight=args.distill_weight,
            Teacher_Model_RGB=teacher_rgb_model,
            Teacher_Model_IR=teacher_ir_model,
            Teacher_Model_RGB_Path=str(teacher_rgb_path),
            Teacher_Model_IR_Path=str(teacher_ir_path),
            loss_type=args.loss_type,
            online=args.online,
            distill_student_rgb_layers=student_cfg["layers_rgb"],
            distill_student_ir_layers=student_cfg["layers_ir"],
            distill_teacher_rgb_layers=teacher_cfg_rgb["layers"],
            distill_teacher_ir_layers=teacher_cfg_ir["layers"],
            distill_student_rgb_channels=student_cfg["channels_rgb"],
            distill_student_ir_channels=student_cfg["channels_ir"],
            distill_student_fusion_layers=student_cfg["layers_fusion"],
            distill_student_fusion_channels=student_cfg["channels_fusion"],
            distill_teacher_rgb_channels=teacher_cfg_rgb["channels"],
            distill_teacher_ir_channels=teacher_cfg_ir["channels"],
        )
    if effective_patience is not None:
        overrides["patience"] = effective_patience
    if effective_save_period is not None:
        overrides["save_period"] = effective_save_period

    DEFAULT_CFG.save_dir = str(save_dir)
    trainer = OBBTrainer(overrides=overrides)
    trainer.train()
    if args.audit_mode and not os.environ.get("SEMD_AUDIT_PROBE_CHILD"):
        trainer.run_audit_probes()


if __name__ == "__main__":
    main()
