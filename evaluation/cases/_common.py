"""評価ケース共通のヘルパ: scope/envelope・None安全比較・条件別の集計と判定。"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from evaluation import config
from evaluation.core import metrics


def ge(value: Optional[float], minimum: float) -> bool:
    """None 安全な「value ≥ minimum」。value が None なら False。"""
    return value is not None and value >= minimum


def scope(ctx, iou=config.IOU_PRIMARY, conf=config.OPERATING_CONF) -> Dict:
    return {
        "model": config.MODEL_NAME,
        "device": ctx.device,
        "dataset": "COCO val2017",
        "num_images": ctx.num_images_evaluated,
        "target_classes": [config.COCO_ID_TO_NAME[c] for c in config.TARGET_COCO_IDS],
        "operating_conf": conf,
        "iou_match": iou,
        "iscrowd": "excluded",
    }


def envelope(case_id, title, mandatory, ctx, iou=config.IOU_PRIMARY, conf=config.OPERATING_CONF) -> Dict:
    return {
        "case_id": case_id,
        "title": title,
        "mandatory": mandatory,
        "executed": True,
        "scope": scope(ctx, iou=iou, conf=conf),
        "metrics": {},
        "criteria": {},
        "result": None,
        "flags": [],
    }


def axis_gate(recall: Optional[float], precision: Optional[float]) -> Dict:
    """条件別2軸（再現率≥0.80・適合率≥0.70）のゲート内訳。"""
    return {
        "recall": {"value": recall, "min": config.COND_RECALL_MIN, "pass": ge(recall, config.COND_RECALL_MIN)},
        "precision": {"value": precision, "min": config.COND_PRECISION_MIN, "pass": ge(precision, config.COND_PRECISION_MIN)},
    }


def condition_result(by_bin: Dict, evaluated: List[str]) -> str:
    """条件別ケースの総合 result。成立サブセットが全て2軸合格なら PASS、一つでも未達で FAIL。

    成立サブセットが一つも無ければ NOT_EXECUTED。
    """
    if not evaluated:
        return "NOT_EXECUTED"
    for b in evaluated:
        if not (ge(by_bin[b]["recall"], config.COND_RECALL_MIN)
                and ge(by_bin[b]["precision"], config.COND_PRECISION_MIN)):
            return "FAIL"
    return "PASS"


def evaluate_image_groups(ctx, img_group: Dict[int, str], bins: List[str]):
    """画像単位サブセット（混雑度・明度）の集計。

    img_group: {iid: bin}（全評価画像を分類済み。None は除外）。
    戻り値: (by_bin, evaluated_bins)。under_min は画像数 < MIN_SUBSET で判定。
    """
    img_count = Counter(g for g in img_group.values() if g is not None)
    group_recs: Dict[str, List[Dict]] = {b: [] for b in bins}
    for t in ctx.op_records:
        g = img_group.get(t["image_id"])
        if g is not None:
            group_recs[g].append(t["rec"])
    by_bin: Dict[str, Dict] = {}
    evaluated: List[str] = []
    for b in bins:
        m = metrics.summarize(group_recs[b])
        n = int(img_count.get(b, 0))
        under = n < config.MIN_SUBSET
        by_bin[b] = {
            "recall": m["recall"], "precision": m["precision"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "num_images": n, "under_min": under,
        }
        if not under:
            evaluated.append(b)
    return by_bin, evaluated
