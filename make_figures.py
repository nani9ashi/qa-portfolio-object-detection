#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results/ の評価成果物から README 用の図を再生成する。

すべての数値・基準値は results/ の JSON / CSV から読み込み、図側にハードコードしない。
これにより「図は results/ から再生成可能（手描きではない）」という監査可能性を担保する。

使い方:
    python make_figures.py                         # results/ -> docs/images/
    python make_figures.py --results results --out docs/images
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

try:
    import japanize_matplotlib  # noqa: F401  日本語フォント登録
except Exception:
    # フォールバック: 環境に実在する CJK フォントを選ぶ（Windows の Yu Gothic 等も含む）
    from matplotlib import font_manager
    _avail = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic",
                 "Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic"]:
        if cand in _avail:
            matplotlib.rcParams["font.family"] = cand
            break

# ---- house style ---------------------------------------------------------
PASS_GREEN = "#2E7D32"
FAIL_RED = "#C62828"
FAIL_EDGE = "#7F1414"
PREC_BLUE = "#1565C0"
GT_GRAY = "#90A4AE"
LINE_DARK = "#37474F"
TEXT = "#263238"
OP_ORANGE = "#EF6C00"
GRID = "#CFD8DC"

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
    "axes.edgecolor": "#90A4AE",
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "font.size": 11,
})

JP = {"small": "小", "medium": "中", "large": "大", "low": "低", "high": "高"}


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return f"{x*100:.0f}%"


# ---- Figure 1: 全体 PR 曲線 ----------------------------------------------
def fig_pr_curve(results: Path, out: Path):
    rows = []
    with open(results / "eval_007_pr_curve.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    rows.sort(key=lambda d: d["conf_threshold"])

    overall = load_json(results / "eval_001_overall.json")
    target = overall["criteria"]["recall_min"]                 # 0.95
    op_conf = overall["scope"]["operating_conf"]               # 0.25
    op_rec = overall["metrics"]["recall"]
    op_prec = overall["metrics"]["precision"]

    rec = [d["recall"] for d in rows]
    prec = [d["precision"] for d in rows]
    confs = [d["conf_threshold"] for d in rows]
    max_rec = max(rec)
    max_rec_conf = confs[rec.index(max_rec)]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    # 到達不能領域（再現率 >= 基準）を薄く塗る
    ax.axvspan(target, 1.0, color=FAIL_RED, alpha=0.06, zorder=0)
    ax.axvline(target, color=FAIL_RED, ls="--", lw=1.6, zorder=2)
    ax.text(target - 0.012, 0.06, f"合格基準  再現率 {target:.2f}",
            color=FAIL_RED, rotation=90, va="bottom", ha="right", fontsize=10)

    ax.plot(rec, prec, "-o", color=PREC_BLUE, lw=2, ms=5, zorder=3,
            label="PR 曲線（信頼度を 0.05〜0.95 で掃引）")

    # 運用点
    ax.scatter([op_rec], [op_prec], s=130, color=OP_ORANGE, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate(f"運用点  conf {op_conf:.2f}\n再現率 {op_rec:.2f} / 適合率 {op_prec:.2f}",
                xy=(op_rec, op_prec), xycoords="data",
                xytext=(0.40, 0.50), textcoords="axes fraction",
                ha="left", fontsize=10, color=TEXT,
                arrowprops=dict(arrowstyle="->", color=OP_ORANGE, lw=1.4))

    # 再現率の上限（最小信頼度側）
    ax.annotate(f"再現率の上限 ≈ {max_rec:.2f}（conf {max_rec_conf:.2f} まで下げても）\n"
                f"→ 閾値調整では基準 {target:.2f} に届かない",
                xy=(max_rec, prec[rec.index(max_rec)]), xycoords="data",
                xytext=(0.03, 0.17), textcoords="axes fraction",
                ha="left", fontsize=10, color=FAIL_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=FAIL_RED, lw=1.4))

    ax.set_xlabel("再現率（見逃しの少なさ） Recall")
    ax.set_ylabel("適合率（誤検出の少なさ） Precision")
    ax.set_title("全体 PR 曲線 — 閾値調整では再現率 0.95 に到達しない\n"
                 "対象 5 クラス・IoU 0.5・COCO val2017 全 5000 枚", loc="left")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    fig.savefig(out / "01_overall_pr_curve.png")
    plt.close(fig)
    print(f"[fig1] max recall={max_rec:.4f} at conf={max_rec_conf}  -> 01_overall_pr_curve.png")


# ---- Figure 2: 条件別 再現率・適合率 --------------------------------------
def _bar_panel(ax, buckets, data, rec_min, prec_min, title):
    import numpy as np
    x = np.arange(len(buckets))
    w = 0.38
    for i, b in enumerate(buckets):
        r = data[b]["recall"]
        p = data[b]["precision"]
        r_pass = r >= rec_min
        p_pass = p >= prec_min
        # recall bar
        ax.bar(x[i] - w/2, r, w,
               color=(PASS_GREEN if r_pass else FAIL_RED),
               edgecolor=("none" if r_pass else FAIL_EDGE),
               hatch=("" if r_pass else "///"), zorder=3)
        # precision bar
        ax.bar(x[i] + w/2, p, w, color=PREC_BLUE,
               edgecolor=(FAIL_RED if not p_pass else "none"),
               linewidth=(2.0 if not p_pass else 0), zorder=3)
        ax.text(x[i] - w/2, r + 0.02, f"{r:.2f}", ha="center", va="bottom",
                fontsize=9, color=(FAIL_RED if not r_pass else TEXT),
                fontweight=("bold" if not r_pass else "normal"))
        ax.text(x[i] + w/2, p + 0.02, f"{p:.2f}", ha="center", va="bottom",
                fontsize=9, color=TEXT)

    ax.axhline(rec_min, color=LINE_DARK, ls="-", lw=1.3, zorder=2)
    ax.axhline(prec_min, color=LINE_DARK, ls=":", lw=1.3, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([JP.get(b, b) for b in buckets])
    ax.set_ylim(0, 1.12)
    ax.set_title(title)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_conditions(results: Path, out: Path):
    size = load_json(results / "eval_004_size.json")
    cong = load_json(results / "eval_005_congestion.json")
    brt = load_json(results / "eval_006_brightness.json")
    rec_min = size["criteria"]["recall_min"]      # 0.8
    prec_min = size["criteria"]["precision_min"]  # 0.7

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    _bar_panel(axes[0], ["small", "medium", "large"], size["metrics"]["by_size"],
               rec_min, prec_min, "物体サイズ別")
    _bar_panel(axes[1], ["low", "medium", "high"], cong["metrics"]["by_congestion"],
               rec_min, prec_min, "混雑度別")
    _bar_panel(axes[2], ["low", "medium", "high"], brt["metrics"]["by_brightness"],
               rec_min, prec_min, "明度別")
    axes[0].set_ylabel("スコア")

    legend = [
        Patch(facecolor=PASS_GREEN, label="再現率（基準クリア）"),
        Patch(facecolor=FAIL_RED, hatch="///", edgecolor=FAIL_EDGE, label="再現率（基準未達）"),
        Patch(facecolor=PREC_BLUE, label="適合率"),
        Line2D([0], [0], color=LINE_DARK, ls="-", lw=1.3, label=f"再現率の基準 {rec_min:.2f}"),
        Line2D([0], [0], color=LINE_DARK, ls=":", lw=1.3, label=f"適合率の基準 {prec_min:.2f}"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=9.5)
    fig.suptitle("条件別の品質 — 失敗は再現率（見逃し）に集中し、明度では変わらない",
                 y=1.14, fontsize=13, fontweight="bold", x=0.5)
    fig.savefig(out / "02_recall_precision_by_condition.png")
    plt.close(fig)
    print("[fig2] -> 02_recall_precision_by_condition.png")


# ---- Figure 3: 見逃し（FN）の集中：サイズ別 -------------------------------
def fig_miss_breakdown(results: Path, out: Path):
    import numpy as np
    size = load_json(results / "eval_004_size.json")["metrics"]["by_size"]
    overall = load_json(results / "eval_001_overall.json")
    total_fn = overall["metrics"]["fn"]
    rec_overall = overall["metrics"]["recall"]

    order = ["small", "medium", "large"]
    gt = [size[b]["gt_count"] for b in order]
    fn = [size[b]["fn"] for b in order]
    labels = [JP[b] for b in order]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w/2, gt, w, color=GT_GRAY, zorder=3, label="正解(GT)総数")
    bars_fn = ax.bar(x + w/2, fn, w, color=FAIL_RED, edgecolor=FAIL_EDGE,
                     hatch="///", zorder=3, label="見逃し(FN)")

    for i, b in enumerate(order):
        ax.text(x[i] - w/2, gt[i] + 60, f"{gt[i]:,}", ha="center", va="bottom",
                fontsize=9, color=TEXT)
        ax.text(x[i] + w/2, fn[i] + 60, f"{fn[i]:,}\n({pct(fn[i]/total_fn)})",
                ha="center", va="bottom", fontsize=9.5, color=FAIL_RED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("物体サイズ")
    ax.set_ylabel("件数")
    ax.set_ylim(0, max(gt) * 1.22)
    ax.set_title(f"見逃し（FN）の集中 — 全 {total_fn:,} 件の見逃しの大半が小物体\n"
                 f"（全体再現率 {rec_overall:.2f}。小物体は GT 最多かつ FN 最多）", loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(out / "03_miss_breakdown_by_size.png")
    plt.close(fig)
    print(f"[fig3] total_fn={total_fn}, small share={fn[0]/total_fn:.3f}  -> 03_miss_breakdown_by_size.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--out", default="docs/images", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig_pr_curve(args.results, args.out)
    fig_conditions(args.results, args.out)
    fig_miss_breakdown(args.results, args.out)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
