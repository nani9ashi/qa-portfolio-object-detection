"""総合判定（AND）と summary.json 生成、整合性アサート。

- 合格水準が改変（緩め）られていないかを検証する。
- 出力に F1 等の統合スコアが混入していないかを検証する（test-plan §2.4）。
- 必須ケースの AND 判定で導入可否を出す。NOT_EXECUTED（成立しない条件別）は AND から除外。
  ERROR の必須ケースは「合格と証明できない」ため No-Go 扱い。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict

from evaluation import config

JST = timezone(timedelta(hours=9))


def assert_criteria_not_loosened() -> None:
    actual = {k: getattr(config, k) for k in config.EXPECTED_CRITERIA}
    if actual != config.EXPECTED_CRITERIA:
        raise AssertionError(
            f"合格水準が改変されています（実測に合わせて緩めてはいけない）: "
            f"{actual} != {config.EXPECTED_CRITERIA}"
        )


def _assert_no_f1(obj, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() == "f1":
                raise AssertionError(f"統合スコア f1 が出力に混入しています: {path}.{k}")
            _assert_no_f1(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_f1(v, f"{path}[{i}]")


def _failed_subcriteria(case: dict) -> str:
    gate = case.get("gate", {})
    fails = []
    for k, v in gate.items():
        if isinstance(v, dict) and "pass" in v:          # 001 の軸 / 008 の時間
            if not v["pass"]:
                fails.append(k)
        elif isinstance(v, dict):                          # 条件別: gate[bin] = {recall:{pass}, precision:{pass}}
            for axis, av in v.items():
                if isinstance(av, dict) and not av.get("pass", True):
                    fails.append(f"{k}.{axis}")
    return ", ".join(fails)


def determine(cases: Dict[str, dict]) -> dict:
    mand = {cid: r for cid, r in cases.items() if r.get("mandatory")}
    errored = [cid for cid, r in mand.items() if r["result"] == "ERROR"]
    not_exec = [cid for cid, r in mand.items() if r["result"] == "NOT_EXECUTED"]
    considered = [cid for cid, r in mand.items() if r["result"] in ("PASS", "FAIL")]
    failing = [cid for cid in considered if mand[cid]["result"] == "FAIL"]

    go = (not failing) and (not errored) and bool(considered)
    determination = "導入可 (Go)" if go else "導入不可 (No-Go)"

    if go:
        basis = "全必須ケースが合格（AND判定）"
    else:
        parts = []
        for cid in failing:
            sub = _failed_subcriteria(mand[cid])
            parts.append(f"{cid}({sub})" if sub else cid)
        parts += [f"{cid}(ERROR)" for cid in errored]
        basis = "AND判定: 不合格 = [" + ", ".join(parts) + "]"

    return {
        "overall_determination": determination,
        "determination_basis": basis,
        "mandatory_cases_considered": considered,
        "mandatory_failing": failing,
        "mandatory_errored": errored,
        "mandatory_not_executed": not_exec,
    }


def _condense(r: dict) -> dict:
    out = {"mandatory": r.get("mandatory"), "executed": r.get("executed"), "result": r.get("result")}
    if "gate" in r:
        out["gate"] = r["gate"]
    if r.get("flags"):
        out["flags"] = r["flags"]
    return out


def write_summary(cases: Dict[str, dict], environment: dict) -> dict:
    assert_criteria_not_loosened()
    for r in cases.values():
        _assert_no_f1(r)

    det = determine(cases)
    summary = {
        "run_at": datetime.now(JST).isoformat(timespec="seconds"),
        "environment": environment,
        "criteria": config.EXPECTED_CRITERIA,
        "cases": {cid: _condense(r) for cid, r in cases.items()},
        **det,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
