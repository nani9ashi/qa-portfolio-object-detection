"""EVAL-001〜008 を実行する単一エントリ。

データ確保 → 推論キャッシュ（1回）→ 全ケース → results/ に構造化出力 + summary.json。
推論は決定的（test-design §5.2）。合格水準は緩めない（summary が検知）。
git 操作（commit/push）は行わない（ユーザーが実施）。

実行: python evaluation\run_eval.py --num-images 5000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

# repo root を import パスに追加（`python evaluation\run_eval.py` でも動くように）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import config, summary                                  # noqa: E402
from evaluation.core import brightness, cache, data, gt, inference, matching  # noqa: E402
from evaluation.core.context import EvalContext                          # noqa: E402
from evaluation.cases import (                                           # noqa: E402
    eval_001_overall, eval_002_map, eval_003_per_class, eval_004_size,
    eval_005_congestion, eval_006_brightness, eval_007_prcurve, eval_008_timing,
)

JST = timezone(timedelta(hours=9))

CASES = [
    ("EVAL-001", eval_001_overall, "eval_001_overall.json"),
    ("EVAL-002", eval_002_map, "eval_002_map.json"),
    ("EVAL-003", eval_003_per_class, "eval_003_per_class.json"),
    ("EVAL-004", eval_004_size, "eval_004_size.json"),
    ("EVAL-005", eval_005_congestion, "eval_005_congestion.json"),
    ("EVAL-006", eval_006_brightness, "eval_006_brightness.json"),
    ("EVAL-007", eval_007_prcurve, "eval_007_prcurve.json"),
    ("EVAL-008", eval_008_timing, "eval_008_timing.json"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def _print_final(results: dict, s: dict) -> None:
    log("")
    log("#" * 72)
    log(f"# 総合判定: {s['overall_determination']}")
    log(f"# 根拠: {s['determination_basis']}")
    log("#" * 72)
    for cid, _mod, _f in CASES:
        r = results[cid]
        tag = "必須" if r.get("mandatory") else "参考"
        log(f"  {cid} [{tag}] {r['result']}")
        g = r.get("gate")
        if g and cid == "EVAL-001":
            for axis, gv in g.items():
                mark = "OK" if gv["pass"] else "NG"
                log(f"      {axis}: {gv['value']} (>= {gv['min']}) [{mark}]")
        elif g and cid in ("EVAL-004", "EVAL-005", "EVAL-006"):
            for b, axes in g.items():
                rc, pr = axes["recall"], axes["precision"]
                log(f"      {b}: recall={rc['value']}(>= {rc['min']}) precision={pr['value']}(>= {pr['min']})")
        elif g and cid == "EVAL-008":
            gv = g["mean_ms_per_frame"]
            mark = "OK" if gv["pass"] else "NG"
            log(f"      mean={gv['value']:.2f}ms (<= {gv['max']}ms) [{mark}]")
    log("")
    log(f"  詳細: {config.RESULTS_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOv8n 導入可否評価 (EVAL-001〜008)")
    ap.add_argument("--num-images", type=int, default=5000, help="評価画像数（最小1000を推奨）")
    ap.add_argument("--device", default="cuda", help="推論デバイス（cuda/cpu）。EVAL-008 は常に GPU")
    ap.add_argument("--no-cache", action="store_true", help="検出キャッシュを使わず再推論する")
    args = ap.parse_args()

    summary.assert_criteria_not_loosened()  # 起動時に基準改変を検知
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log(f"YOLOv8n 導入可否評価: {args.num_images}枚 / device={args.device}")
    log("=" * 72)

    coco, img_ids, paths = data.prepare(args.num_images, log)
    if not img_ids:
        log("[error] 評価対象画像が0枚です。中止します。")
        sys.exit(1)

    gt_by, cong = gt.load_gt(coco, img_ids)

    detections, img_meta = (None, None) if args.no_cache else cache.load_if_valid(img_ids)
    if detections is None:
        log(f"[infer] 推論を実行（低conf={config.INFER_CONF}, device={args.device}）…")
        model = inference.load_model()
        detections = inference.run_inference(model, paths, args.device, log)
        log("[meta] 画像の平均輝度を計算…")
        img_meta = {
            iid: {"mean_luma": brightness.mean_luminance(paths[iid]),
                  "congestion_count": cong.get(iid, 0)}
            for iid in img_ids
        }
        cache.save(detections, img_meta, img_ids)
        log("[cache] 検出キャッシュを保存")
    else:
        log("[cache] 既存キャッシュを再利用（推論スキップ）")

    op_records = matching.build_records(
        detections, gt_by, img_ids, config.OPERATING_CONF, config.IOU_PRIMARY)
    ctx = EvalContext(
        coco=coco, img_ids=img_ids, img_paths=paths, detections=detections,
        gt_by_img_cls=gt_by, congestion_count=cong, img_meta=img_meta,
        device=args.device, op_records=op_records,
    )

    run_at = datetime.now(JST).isoformat(timespec="seconds")
    results = {}
    for cid, mod, fname in CASES:
        log(f"[run] {cid} …")
        r = mod.run(ctx)
        r["run_at"] = run_at
        (config.RESULTS_DIR / fname).write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        results[cid] = r

    import torch
    environment = {
        "device": args.device,
        "num_images": ctx.num_images_evaluated,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }
    s = summary.write_summary(results, environment)
    _print_final(results, s)


if __name__ == "__main__":
    main()
