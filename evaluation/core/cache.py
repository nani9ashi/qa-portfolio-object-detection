"""検出キャッシュと画像メタの永続化（parquet）。manifest で無効化を判定する。

「推論は1回だけ」を担保する。manifest（モデル/conf/iou/imgsz/画像集合のハッシュ）が一致すれば
推論を再実行せずキャッシュをロードする。ケース側のコード修正はキャッシュを無効化しない。
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import polars as pl

from evaluation import config

DET_FILE = config.CACHE_DIR / "detections.parquet"
META_FILE = config.CACHE_DIR / "image_meta.parquet"
MANIFEST = config.CACHE_DIR / "manifest.json"

_DET_SCHEMA = {
    "image_id": pl.Int64, "x1": pl.Float64, "y1": pl.Float64, "x2": pl.Float64,
    "y2": pl.Float64, "score": pl.Float64, "cls": pl.Int64, "area": pl.Float64,
}
_META_SCHEMA = {"image_id": pl.Int64, "mean_luma": pl.Float64, "congestion_count": pl.Int64}


def _img_ids_hash(img_ids) -> str:
    h = hashlib.sha256()
    for i in img_ids:
        h.update(f"{int(i)},".encode())
    return h.hexdigest()[:16]


def _model_hash() -> str:
    if config.MODEL_PATH.exists():
        return hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()[:16]
    return "none"


def manifest_key(img_ids) -> Dict:
    return {
        "model": config.MODEL_NAME,
        "model_sha": _model_hash(),
        "infer_conf": config.INFER_CONF,
        "nms_iou": config.NMS_IOU,
        "imgsz": config.IMGSZ,
        "max_det": config.MAX_DET,
        "classes": config.TARGET_YOLO_IDS,
        "num_images": len(img_ids),
        "img_ids_sha": _img_ids_hash(img_ids),
    }


def load_if_valid(img_ids) -> Tuple[Optional[Dict], Optional[Dict]]:
    """manifest が一致すれば (detections, img_meta) を返す。なければ (None, None)。"""
    if not (MANIFEST.exists() and DET_FILE.exists() and META_FILE.exists()):
        return None, None
    try:
        saved = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if saved != manifest_key(img_ids):
        return None, None
    det_df = pl.read_parquet(DET_FILE)
    detections: Dict[int, List[Dict]] = {int(i): [] for i in img_ids}
    for row in det_df.iter_rows(named=True):
        detections.setdefault(int(row["image_id"]), []).append({
            "score": row["score"],
            "box": [row["x1"], row["y1"], row["x2"], row["y2"]],
            "cls": int(row["cls"]),
            "area": row["area"],
        })
    meta_df = pl.read_parquet(META_FILE)
    img_meta = {
        int(r["image_id"]): {"mean_luma": r["mean_luma"], "congestion_count": int(r["congestion_count"])}
        for r in meta_df.iter_rows(named=True)
    }
    return detections, img_meta


def save(detections: Dict[int, List[Dict]], img_meta: Dict[int, Dict], img_ids) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for iid, dets in detections.items():
        for d in dets:
            b = d["box"]
            rows.append({
                "image_id": int(iid), "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                "score": d["score"], "cls": int(d["cls"]), "area": d["area"],
            })
    pl.DataFrame(rows, schema=_DET_SCHEMA).write_parquet(DET_FILE)
    meta_rows = [
        {"image_id": int(i), "mean_luma": img_meta[i]["mean_luma"],
         "congestion_count": int(img_meta[i]["congestion_count"])}
        for i in img_meta
    ]
    pl.DataFrame(meta_rows, schema=_META_SCHEMA).write_parquet(META_FILE)
    MANIFEST.write_text(json.dumps(manifest_key(img_ids), ensure_ascii=False, indent=2), encoding="utf-8")
