"""EVAL-006（成立時必須・AC-4a/AC-4b）: 明度別の再現率・適合率（画像単位）。

画像のグレースケール平均輝度で 低<80 / 中80-159 / 高≥160 に分類。
成立条件（test-design §4.3）: 「低」明度サブセットが30枚以上のときのみ実施。未満なら NOT_EXECUTED
（＝総合のAND判定から除外）。各サブセットの最小要件30枚も適用（未満binはゲート外）。
合格（成立時）= 各明度で 再現率≥0.80 かつ 適合率≥0.70。
"""
from __future__ import annotations

from evaluation import config
from evaluation.cases import _common


def run(ctx) -> dict:
    env = _common.envelope("EVAL-006", "明度別の再現率・適合率", True, ctx)

    img_group = {}
    for iid in ctx.img_ids:
        luma = ctx.img_meta.get(iid, {}).get("mean_luma")
        img_group[iid] = config.brightness_bin(luma) if luma is not None else None

    by_bin, evaluated = _common.evaluate_image_groups(ctx, img_group, config.BRIGHTNESS_BINS)
    env["criteria"] = {
        "recall_min": config.COND_RECALL_MIN, "precision_min": config.COND_PRECISION_MIN,
        "min_unit": "images", "min_subset": config.MIN_SUBSET,
    }

    low_n = by_bin["low"]["num_images"]
    if low_n < config.MIN_SUBSET:
        # 成立条件を満たさない: 本ケースは不実施（AND から除外）
        env["executed"] = False
        env["result"] = "NOT_EXECUTED"
        env["metrics"] = {
            "by_brightness": by_bin,
            "reason": f"low-brightness subset has {low_n} images (< {config.MIN_SUBSET})",
            "low_count": low_n,
        }
        env["flags"] = ["brightness_subset_not_viable"]
        return env

    env["metrics"] = {"by_brightness": by_bin}
    env["result"] = _common.condition_result(by_bin, evaluated)
    env["gate"] = {b: _common.axis_gate(by_bin[b]["recall"], by_bin[b]["precision"]) for b in evaluated}
    return env
