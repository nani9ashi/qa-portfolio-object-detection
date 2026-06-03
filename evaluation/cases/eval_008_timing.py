"""EVAL-008（必須・AC-5）: 1フレームあたりの推論時間（GPU）。

GPU 必須（CPU への無言フォールバックは禁止＝ERROR）。ウォームアップ10フレームを除外し、
cuda.synchronize() で挟んで1フレームずつ計測した定常状態の平均で判定。合格 = 平均 ≤100ms。
"""
from __future__ import annotations

import time

import numpy as np

from evaluation import config
from evaluation.cases import _common
from evaluation.core.inference import load_model

WARMUP = 10
SAMPLE = 300  # 計測対象フレーム数の上限（ウォームアップ込み）


def run(ctx) -> dict:
    env = _common.envelope("EVAL-008", "推論時間（GPU・1フレーム）", True, ctx, conf=config.OPERATING_CONF, iou=None)
    env["criteria"] = {"infer_time_max_ms": config.INFER_TIME_MAX_MS}

    import torch
    if not torch.cuda.is_available():
        env["executed"] = False
        env["result"] = "ERROR"
        env["metrics"] = {"message": "AC-5 requires GPU but torch.cuda.is_available() == False"}
        env["flags"] = ["gpu_unavailable"]
        return env

    device = "cuda"
    model = load_model()
    ids = ctx.img_ids[: min(SAMPLE, len(ctx.img_ids))]
    paths = [str(ctx.img_paths[i]) for i in ids]

    # ウォームアップ（CUDA初期化・autotune を計測から除外）
    for p in paths[:WARMUP]:
        model.predict(source=p, imgsz=config.IMGSZ, conf=config.OPERATING_CONF, device=device, verbose=False)
    torch.cuda.synchronize()

    times = []
    for p in paths[WARMUP:]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(source=p, imgsz=config.IMGSZ, conf=config.OPERATING_CONF, device=device, verbose=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    if not times:
        env["executed"] = False
        env["result"] = "ERROR"
        env["metrics"] = {"message": "insufficient frames for steady-state timing"}
        return env

    mean_ms = float(np.mean(times))
    env["metrics"] = {
        "mean_ms_per_frame": mean_ms,
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "warmup_frames": WARMUP,
        "timed_frames": len(times),
        "device": torch.cuda.get_device_name(0),
    }
    passed = mean_ms <= config.INFER_TIME_MAX_MS
    env["result"] = "PASS" if passed else "FAIL"
    env["gate"] = {"mean_ms_per_frame": {"value": mean_ms, "max": config.INFER_TIME_MAX_MS, "pass": passed}}
    return env
