#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_overlays.py — 検出例オーバーレイ図（④）を生成する。

evaluation/core の GT 取得・検出キャッシュ・マッチングを **そのまま再利用** するため、描画する
TP/FP/FN は評価本体（EVAL-001/004）の集計と原理的に一致する。運用点・対象クラス・iscrowd 除外は
すべて evaluation/config.py 由来。検出キャッシュがあれば推論を省く（GPU 不要）。

選定（決定的・再現可能）:
  - 対象クラスの GT を 1 つ以上含む画像を「FN 件数の降順 → image_id 昇順」で並べ、上位 3 枚
    （最も見逃しが多いフレーム＝失敗の典型）。
  - 対比として TP 最多かつ FN = 0（同点は image_id 昇順）の 1 枚（大きく明瞭なら正しく検出できる例）。

これは「集計値」ではなく「見逃しが実際どう見えるか」を示す **例示**。選定は手作業ではなく上記
ルールで自動化している。

使い方:
    python make_overlays.py                       # cache 再利用 → docs/images/04_detection_overlays.png
    python make_overlays.py --num-images 5000 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

try:
    import japanize_matplotlib  # noqa: F401  日本語フォント登録
except Exception:
    # フォールバック: 環境に実在する CJK フォントを選ぶ（Windows の Yu Gothic 等も含む）
    from matplotlib import font_manager
    _avail = {f.name for f in font_manager.fontManager.ttflist}
    for _cand in ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic",
                  "Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic"]:
        if _cand in _avail:
            matplotlib.rcParams["font.family"] = _cand
            break

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root を import パスへ
from evaluation import config                                          # noqa: E402
from evaluation.core import cache, gt as gt_core, inference, matching  # noqa: E402

# house style（make_figures.py と統一: 緑=TP / 赤=FN / オレンジ=FP）
GREEN, RED, ORANGE = "#2E7D32", "#C62828", "#EF6C00"
plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.dpi": 150, "font.size": 11,
})


def build_per_image(num_images: int, device: str, log=print):
    """評価本体と同じ手順で per-image の TP/FP/FN（box 付き）と画像パスを返す。"""
    from pycocotools.coco import COCO
    coco = COCO(str(config.ANN_FILE))
    img_ids = sorted(coco.getImgIds())[:num_images]
    gt_by, _cong = gt_core.load_gt(coco, img_ids)
    paths = {int(i): config.IMG_DIR / coco.loadImgs(i)[0]["file_name"] for i in img_ids}

    detections, _meta = cache.load_if_valid(img_ids)
    if detections is None:
        log("[overlay] 検出キャッシュが無いため推論を実行（GPU 推奨）…")
        model = inference.load_model()
        detections = inference.run_inference(model, paths, device, log)
    else:
        log("[overlay] 検出キャッシュを再利用（推論なし・評価本体と同一の検出）")

    # 評価本体と同一のマッチング（運用点 conf 0.25 / IoU 0.5）
    tagged = matching.build_records(
        detections, gt_by, img_ids, config.OPERATING_CONF, config.IOU_PRIMARY)
    per_img: dict = {}
    for t in tagged:
        e = per_img.setdefault(int(t["image_id"]), {"tp": [], "fp": [], "fn": []})
        rec = t["rec"]
        if rec["type"] == "TP":
            e["tp"].append(rec["det_box"])
        elif rec["type"] == "FP":
            e["fp"].append(rec["det_box"])
        else:  # FN
            e["fn"].append(rec["gt_box"])
    return per_img, paths


def select(per_img):
    """決定的選定: FN 降順→id 昇順の上位3枚 ＋ FN=0 で TP 最多の対比1枚。"""
    items = [(iid, e) for iid, e in per_img.items() if (len(e["tp"]) + len(e["fn"])) > 0]
    worst = sorted(items, key=lambda kv: (-len(kv[1]["fn"]), kv[0]))[:3]
    zero_fn = [(iid, e) for iid, e in items if len(e["fn"]) == 0]
    contrast = sorted(zero_fn, key=lambda kv: (-len(kv[1]["tp"]), kv[0]))[:1]
    return worst + contrast


def draw_cell(ax, iid, e, img_path):
    img = cv2.imread(str(img_path))
    if img is not None:
        ax.imshow(img[:, :, ::-1])  # BGR -> RGB
    ax.set_xticks([]); ax.set_yticks([])
    for (x1, y1, x2, y2) in e["tp"]:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=GREEN, lw=2))
    for (x1, y1, x2, y2) in e["fp"]:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=ORANGE, lw=1.6, ls="--"))
    for (x1, y1, x2, y2) in e["fn"]:  # 見逃しは最重要 → 最も目立たせる
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=RED, lw=2.6))
    ax.set_title(f"image {iid} — TP {len(e['tp'])} / FP {len(e['fp'])} / FN {len(e['fn'])}",
                 fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=5000)
    ap.add_argument("--device", default="cuda", help="キャッシュが無い場合のみ使用")
    ap.add_argument("--out", default="docs/images/04_detection_overlays.png")
    args = ap.parse_args()

    per_img, paths = build_per_image(args.num_images, args.device)

    # 整合チェック: 全画像合計の TP/FP/FN は EVAL-001 と一致するはず（core 再利用のため自動一致）
    tot = {k: sum(len(e[k]) for e in per_img.values()) for k in ("tp", "fp", "fn")}
    print(f"[overlay] 全体集計 TP={tot['tp']} FP={tot['fp']} FN={tot['fn']}（EVAL-001 と一致するはず）")

    sel = select(per_img)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (iid, e) in zip(axes.ravel(), sel):
        draw_cell(ax, iid, e, paths[iid])

    legend = [Patch(facecolor="none", edgecolor=GREEN, label="検出（TP）"),
              Patch(facecolor="none", edgecolor=RED, label="見逃し（FN）"),
              Patch(facecolor="none", edgecolor=ORANGE, label="誤検出（FP）")]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.suptitle("検出例（運用点 conf 0.25 / IoU 0.5・COCO val2017）— 赤い「見逃し」が実際どう起きているか",
                 y=1.06, fontsize=13, fontweight="bold")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", out, "| selected:",
          [(iid, f"TP{len(e['tp'])}/FP{len(e['fp'])}/FN{len(e['fn'])}") for iid, e in sel])


if __name__ == "__main__":
    main()
