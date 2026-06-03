"""低 conf で1回だけ推論し、検出キャッシュ（画像ごとの box/score/cls/area）を作る。

検出は全 EVAL ケースで再利用される。クラスは推論時に対象5クラスへ絞り、COCO 91-id に変換。
det の area は bbox 面積（detection 側のサイズ判定に使用。GT は area フィールドを使用）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from evaluation import config


def load_model():
    """既存の YOLOv8n 重みをロード（推論専用・改変しない）。"""
    from ultralytics import YOLO
    target = str(config.MODEL_PATH) if config.MODEL_PATH.exists() else config.MODEL_NAME
    return YOLO(target)


def infer_image(model, img_path, device: str) -> List[Dict]:
    res = model.predict(
        source=str(img_path), conf=config.INFER_CONF, iou=config.NMS_IOU,
        classes=config.TARGET_YOLO_IDS, imgsz=config.IMGSZ, max_det=config.MAX_DET,
        device=device, verbose=False,
    )[0]
    dets: List[Dict] = []
    boxes = res.boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            dets.append({
                "score": float(conf[i]),
                "box": [x1, y1, x2, y2],
                "cls": int(config.COCO80_TO_91[int(cls[i])]),
                "area": (x2 - x1) * (y2 - y1),
            })
    return dets


def run_inference(model, img_paths: Dict[int, Path], device: str, log=print) -> Dict[int, List[Dict]]:
    """全画像で1回ずつ推論し、{iid: [det, ...]} を返す。"""
    detections: Dict[int, List[Dict]] = {}
    n = len(img_paths)
    for k, (iid, path) in enumerate(img_paths.items(), 1):
        detections[int(iid)] = infer_image(model, path, device)
        if k % 200 == 0 or k == n:
            log(f"  [infer] {k}/{n}")
    return detections
