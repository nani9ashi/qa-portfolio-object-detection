"""EVAL-002（参考・AC-1/AC-2）: mAP@0.5 と mAP@0.5:0.95（pycocotools COCOeval）。

合否対象外。COCOeval は信頼度で切らず低conf全量を投入する（AP は PR曲線の積分のため）。
"""
from __future__ import annotations

import numpy as np

from evaluation import config
from evaluation.cases import _common


def run(ctx) -> dict:
    env = _common.envelope("EVAL-002", "mAP（参考）", False, ctx)
    from pycocotools.cocoeval import COCOeval

    dets = []
    for iid, ds in ctx.detections.items():
        for d in ds:
            x1, y1, x2, y2 = d["box"]
            dets.append({
                "image_id": int(iid), "category_id": int(d["cls"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d["score"],
            })

    if not dets:
        env["metrics"] = {"overall": {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0}, "per_class": {}}
        env["result"] = "REFERENCE"
        return env

    coco_dt = ctx.coco.loadRes(dets)
    ev = COCOeval(ctx.coco, coco_dt, iouType="bbox")
    ev.params.imgIds = [int(i) for i in ctx.img_ids]
    ev.params.catIds = config.TARGET_COCO_IDS
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    prec = ev.eval["precision"]  # [T(iou), R(recall), K(cat), A(area), M(maxDet)]
    overall = {"mAP@0.5": float(ev.stats[1]), "mAP@0.5:0.95": float(ev.stats[0])}
    per_class = {}
    for k, cid in enumerate(config.TARGET_COCO_IDS):
        p50 = prec[0, :, k, 0, 2]
        pall = prec[:, :, k, 0, 2]
        ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else 0.0
        ap = float(np.mean(pall[pall > -1])) if (pall > -1).any() else 0.0
        per_class[config.COCO_ID_TO_NAME[cid]] = {"AP@0.5": ap50, "AP@0.5:0.95": ap}

    env["metrics"] = {"overall": overall, "per_class": per_class}
    env["result"] = "REFERENCE"
    return env
