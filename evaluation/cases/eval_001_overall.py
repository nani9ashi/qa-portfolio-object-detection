"""EVAL-001（必須・AC-1/AC-3/AC-2）: 全体の再現率・適合率・平均IoU。

運用点 conf0.25 / IoU0.5。合格 = 再現率≥0.95 かつ 適合率≥0.70 かつ 平均IoU≥0.7（AND）。
IoU0.75 を補助記録（参考・ゲート外）。
"""
from __future__ import annotations

from evaluation import config
from evaluation.core import matching, metrics
from evaluation.cases import _common


def run(ctx) -> dict:
    env = _common.envelope("EVAL-001", "全体の再現率・適合率・平均IoU", True, ctx)

    recs = [t["rec"] for t in ctx.op_records]
    m = metrics.summarize(recs)

    recs75 = [t["rec"] for t in matching.build_records(
        ctx.detections, ctx.gt_by_img_cls, ctx.img_ids, config.OPERATING_CONF, config.IOU_AUX)]
    m75 = metrics.summarize(recs75)

    env["metrics"] = {**m, "aux_iou75": m75}
    env["criteria"] = {
        "recall_min": config.OVERALL_RECALL_MIN,
        "precision_min": config.OVERALL_PRECISION_MIN,
        "mean_iou_min": config.MEAN_IOU_MIN,
    }
    gate = {
        "recall": {"value": m["recall"], "min": config.OVERALL_RECALL_MIN,
                   "pass": _common.ge(m["recall"], config.OVERALL_RECALL_MIN)},
        "precision": {"value": m["precision"], "min": config.OVERALL_PRECISION_MIN,
                      "pass": _common.ge(m["precision"], config.OVERALL_PRECISION_MIN)},
        "mean_iou_tp": {"value": m["mean_iou_tp"], "min": config.MEAN_IOU_MIN,
                        "pass": _common.ge(m["mean_iou_tp"], config.MEAN_IOU_MIN)},
    }
    env["gate"] = gate
    env["result"] = "PASS" if all(g["pass"] for g in gate.values()) else "FAIL"
    return env
