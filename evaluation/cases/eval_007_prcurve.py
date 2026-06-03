"""EVAL-007（参考・AC-1/AC-3）: PR曲線。

信頼度しきい値を 0.05〜0.95（0.05刻み）で動かし、各点の全体 再現率・適合率を算出して CSV 出力。
合否対象外。EVAL-001 不合格時に「運用点変更で合格可能か」を検討する材料。
"""
from __future__ import annotations

import csv

from evaluation import config
from evaluation.core import matching, metrics
from evaluation.cases import _common


def run(ctx) -> dict:
    env = _common.envelope("EVAL-007", "PR曲線（参考）", False, ctx)

    points = []
    for thr in config.PR_THRESHOLDS:
        recs = [t["rec"] for t in matching.build_records(
            ctx.detections, ctx.gt_by_img_cls, ctx.img_ids, thr, config.IOU_PRIMARY)]
        m = metrics.summarize(recs)
        points.append({
            "conf_threshold": thr, "recall": m["recall"], "precision": m["precision"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        })

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.RESULTS_DIR / "eval_007_pr_curve.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conf_threshold", "recall", "precision", "tp", "fp", "fn"])
        for p in points:
            w.writerow([p["conf_threshold"], p["recall"], p["precision"], p["tp"], p["fp"], p["fn"]])

    env["metrics"] = {"csv": "results/eval_007_pr_curve.csv", "points": len(points)}
    env["result"] = "REFERENCE"
    return env
