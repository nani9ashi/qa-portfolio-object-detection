"""TP/FP/FN 記録から再現率・適合率・平均IoU を算出する。

再現率と適合率は独立に並べる。F1 等の統合スコアは意図的に作らない（test-plan §2.4）。
分母が 0（対象/検出が無い）の場合は None（JSON では null）を返し、呼び出し側で flag する。
"""
from __future__ import annotations

from typing import Dict, List, Optional


def count(records: List[Dict]):
    """記録リストから (tp, fp, fn, iou_sum) を集計する。"""
    tp = fp = fn = 0
    iou_sum = 0.0
    for r in records:
        t = r["type"]
        if t == "TP":
            tp += 1
            iou_sum += r["iou"]
        elif t == "FP":
            fp += 1
        else:  # FN
            fn += 1
    return tp, fp, fn, iou_sum


def recall(tp: int, fn: int) -> Optional[float]:
    d = tp + fn
    return (tp / d) if d > 0 else None


def precision(tp: int, fp: int) -> Optional[float]:
    d = tp + fp
    return (tp / d) if d > 0 else None


def mean_iou(iou_sum: float, tp: int) -> Optional[float]:
    return (iou_sum / tp) if tp > 0 else None


def summarize(records: List[Dict]) -> Dict:
    """記録リスト（内側 record の list）から指標 dict を返す。F1 は含めない。"""
    tp, fp, fn, iou_sum = count(records)
    return {
        "recall": recall(tp, fn),
        "precision": precision(tp, fp),
        "mean_iou_tp": mean_iou(iou_sum, tp),
        "tp": tp, "fp": fp, "fn": fn,
    }
