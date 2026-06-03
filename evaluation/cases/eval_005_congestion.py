"""EVAL-005（必須・AC-4a/AC-4b）: 混雑度別の再現率・適合率（画像単位）。

画像内の対象GT数で 低0-2 / 中3-5 / 高6+ に分類（0個は低に含む＝背景での誤検出を見る集合）。
合格 = 各混雑度で 再現率≥0.80 かつ 適合率≥0.70。最小単位は画像数。
"""
from __future__ import annotations

from evaluation import config
from evaluation.cases import _common


def run(ctx) -> dict:
    env = _common.envelope("EVAL-005", "混雑度別の再現率・適合率", True, ctx)

    img_group = {iid: config.congestion_bin(ctx.congestion_count.get(iid, 0)) for iid in ctx.img_ids}
    by_bin, evaluated = _common.evaluate_image_groups(ctx, img_group, config.CONGESTION_BINS)

    env["metrics"] = {"by_congestion": by_bin}
    env["criteria"] = {
        "recall_min": config.COND_RECALL_MIN, "precision_min": config.COND_PRECISION_MIN,
        "min_unit": "images", "min_subset": config.MIN_SUBSET,
    }
    env["result"] = _common.condition_result(by_bin, evaluated)
    env["gate"] = {b: _common.axis_gate(by_bin[b]["recall"], by_bin[b]["precision"]) for b in evaluated}
    return env
