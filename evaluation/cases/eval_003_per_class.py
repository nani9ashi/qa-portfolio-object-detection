"""EVAL-003（参考・AC-1/AC-3）: クラス別の再現率・適合率。

合否対象外。クラス間の偏り（COCO は person が圧倒的多数）を観察し、極端な低下を flag する。
"""
from __future__ import annotations

from collections import defaultdict

from evaluation import config
from evaluation.core import metrics
from evaluation.cases import _common


def run(ctx) -> dict:
    env = _common.envelope("EVAL-003", "クラス別の再現率・適合率（参考）", False, ctx)

    by_cls_recs = defaultdict(list)
    for t in ctx.op_records:
        by_cls_recs[t["cls"]].append(t["rec"])

    per_class = {}
    recalls = []
    for cid in config.TARGET_COCO_IDS:
        m = metrics.summarize(by_cls_recs.get(cid, []))
        per_class[config.COCO_ID_TO_NAME[cid]] = m
        if m["recall"] is not None:
            recalls.append(m["recall"])

    flags = []
    if len(recalls) >= 2 and (max(recalls) - min(recalls)) >= 0.3:
        flags.append("class_recall_disparity")

    env["metrics"] = {"per_class": per_class}
    env["flags"] = flags
    env["result"] = "REFERENCE"
    return env
