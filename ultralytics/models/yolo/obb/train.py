# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBTrainer

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml', epochs=3)
        trainer = OBBTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'obb'
        # super().__init__(cfg, overrides, _callbacks)
        super().__init__(overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        model = OBBModel(cfg, ch=getattr(self.args, 'ch', 6), nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        if self.Distillation is not None and getattr(self.args, 'distill_stage2_cls_only', False):
            base_loss_names = ['box_loss', 'cls_loss', 'dfl_loss']
            if getattr(self, 'FIA', False):
                base_loss_names.append('lif_loss')
            self.loss_names = (
                *base_loss_names,
                'det_loss',
                'cls_kd_loss',
                'num_selected_pos',
                'rgb_selected_ratio',
                'ir_selected_ratio',
                'avg_teacher_reliability',
                'avg_jsd_on_selected',
            )
        elif self.Distillation is not None:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'lif_loss', 'im_loss', 'cm_loss'
        else:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return yolo.obb.OBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
