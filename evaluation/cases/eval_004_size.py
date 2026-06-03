"""EVAL-004（必須・AC-4a/AC-4b）: 物体サイズ別の再現率・適合率（オブジェクト単位）。

マッチングは画像/クラス単位で先に行い、bin はその後に各記録へ付与する（bin先決め→マッチ禁止だと
跨bin対応を歪めるため）。再現率/FN は GT の面積bin、適合率/FP は検出の面積bin で帰属。
合格 = 各サイズ（small/medium/large）で 再現率≥0.80 かつ 適合率≥0.70。最小単位はオブジェクト数。
"""
from __future__ import annotations

from evaluation import config
from evaluation.core import metrics
from evaluation.cases import _common


def run(ctx) -> dict:
    env = _common.envelope("EVAL-004", "物体サイズ別の再現率・適合率", True, ctx)
    bins = config.SIZE_BINS

    tp_gt = {b: 0 for b in bins}    # 再現率側: TP を GT 面積で
    fn = {b: 0 for b in bins}       # 再現率側: FN を GT 面積で
    tp_det = {b: 0 for b in bins}   # 適合率側: TP を 検出 面積で
    fp = {b: 0 for b in bins}       # 適合率側: FP を 検出 面積で
    gt_count = {b: 0 for b in bins}
    det_count = {b: 0 for b in bins}

    for t in ctx.op_records:
        r = t["rec"]
        typ = r["type"]
        if typ == "TP":
            tp_gt[r["gt_size"]] += 1
            gt_count[r["gt_size"]] += 1
            tp_det[r["det_size"]] += 1
            det_count[r["det_size"]] += 1
        elif typ == "FN":
            fn[r["gt_size"]] += 1
            gt_count[r["gt_size"]] += 1
        else:  # FP
            fp[r["det_size"]] += 1
            det_count[r["det_size"]] += 1

    by_size = {}
    evaluated = []
    for b in bins:
        rec = metrics.recall(tp_gt[b], fn[b])
        pre = metrics.precision(tp_det[b], fp[b])
        under = gt_count[b] < config.MIN_SUBSET or det_count[b] < config.MIN_SUBSET
        by_size[b] = {
            "recall": rec, "precision": pre,
            "tp_gt": tp_gt[b], "tp_det": tp_det[b], "fp": fp[b], "fn": fn[b],
            "gt_count": gt_count[b], "det_count": det_count[b], "under_min": under,
        }
        if not under:
            evaluated.append(b)

    env["metrics"] = {"by_size": by_size}
    env["criteria"] = {
        "recall_min": config.COND_RECALL_MIN, "precision_min": config.COND_PRECISION_MIN,
        "min_unit": "objects", "min_subset": config.MIN_SUBSET,
    }
    env["result"] = _common.condition_result(by_size, evaluated)
    env["gate"] = {b: _common.axis_gate(by_size[b]["recall"], by_size[b]["precision"]) for b in evaluated}
    return env
