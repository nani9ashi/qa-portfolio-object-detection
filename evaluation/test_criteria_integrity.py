"""
合格基準の整合性チェック（GPU・評価データ不要、数秒で完了）。

「基準は実測前に固定し、後から緩めない」という方針を CI で継続的に保証する。
- results/summary.json に記録された合格水準が、事前合意の基準（Issue #1 / #2）と完全一致すること
- 合否ゲートに統合スコア（F1 や mAP）が混入していないこと（独立2軸の AND 判定の担保）
を検証する。基準が緩められたり統合スコアが紛れ込んだ瞬間に CI が落ちる。

実行:  pytest -q evaluation/test_criteria_integrity.py
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "results" / "summary.json"

# Issue #1 / #2 で合意した「動かさない」合格水準。ここを緩めると CI が落ちる。
LOCKED_CRITERIA = {
    "OVERALL_RECALL_MIN": 0.95,
    "OVERALL_PRECISION_MIN": 0.70,
    "MEAN_IOU_MIN": 0.70,
    "COND_RECALL_MIN": 0.80,
    "COND_PRECISION_MIN": 0.70,
    "INFER_TIME_MAX_MS": 100.0,
}

# 合否を分けてよい指標（再現率・適合率の独立2軸＋位置精度＋速度）。統合スコアは含めない。
ALLOWED_GATE_METRICS = {"recall", "precision", "mean_iou_tp", "mean_ms_per_frame"}
FORBIDDEN_GATE_METRICS = {"f1", "f1_score", "map", "mAP", "mAP@0.5", "mAP@0.5:0.95"}


def _load_summary():
    if not SUMMARY.exists():
        pytest.fail(
            f"{SUMMARY} が見つかりません。評価を 1 回実行し、summary.json をコミットしてください"
            "（results/summary.json が .gitignore で除外されていないことを確認）。"
        )
    with open(SUMMARY, encoding="utf-8") as f:
        return json.load(f)


def _collect_gate_metrics(node):
    """gate ツリーを再帰的に走査し、合否判定された指標名を収集する。"""
    metrics = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and "pass" in value:
                metrics.add(key)            # 末端の判定ノード。key が指標名
            elif isinstance(value, dict):
                metrics |= _collect_gate_metrics(value)  # bucket 等を再帰
    return metrics


def test_criteria_are_locked():
    """合格水準が事前合意の基準と完全一致する（後から緩めていない）。"""
    summary = _load_summary()
    assert summary.get("criteria") == LOCKED_CRITERIA, (
        "合格水準が固定値から変化しています。基準は実測前に確定し、後から動かさない方針です。"
    )


def test_no_integrated_score_in_gates():
    """合否ゲートに統合スコア（F1 / mAP 等）が混入していない。"""
    summary = _load_summary()
    used = set()
    for case in summary.get("cases", {}).values():
        used |= _collect_gate_metrics(case.get("gate", {}))
    assert used, "gate から判定指標を1つも検出できませんでした（summary.json の構造を確認）。"
    leaked = used & FORBIDDEN_GATE_METRICS
    assert not leaked, f"統合スコアが合否ゲートに混入しています: {sorted(leaked)}"
    unexpected = used - ALLOWED_GATE_METRICS
    assert not unexpected, f"想定外の指標が合否ゲートに使われています: {sorted(unexpected)}"
