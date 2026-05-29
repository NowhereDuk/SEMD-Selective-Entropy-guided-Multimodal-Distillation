# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model='yolov8n-obb.pt', source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'obb'

    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions and keep paired-stream save behavior consistent with detect."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        nc=len(self.model.names),
                                        classes=self.args.classes,
                                        rotated=True)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        f = lambda x: 4 if 4 <= x <= 16 else 4
        results = []
        bs = f(orig_imgs[0].shape[-1])
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i][..., -3:]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape, xywh=True)
            img_path = self.batch[0][i]
            # xywh, r, conf, cls
            obb = torch.cat([pred[:, :4], pred[:, -1:], pred[:, 4:6]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
            pair_path = img_path.replace('images', 'images_ir')
            if orig_imgs[i].shape[-1] >= bs:
                pair_results = []
                pair_img = orig_imgs[i][..., :3]
                pair_results.append(Results(pair_img, path=pair_path, names=self.model.names, obb=obb))
                return results, pair_results
        return results
