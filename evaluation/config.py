"""中央設定: パス・対象クラス・しきい値・サブセット bin・合格水準。

合格水準（criteria）は docs/test-plan.md §5.1（Issue #1/#2 で PdM 承認済み）に対応する。
これは「測るための基準」であり、実測値に合わせて緩めてはならない（summary.py が改変を検知する）。
"""
from __future__ import annotations

from pathlib import Path

# ---- パス（このパッケージ＝repo直下の evaluation/） -------------------------
ROOT = Path(__file__).resolve().parent.parent          # repo root
DATA_DIR = ROOT / "data"                                # 実行用データセット（gitignore）
IMG_DIR = DATA_DIR / "val2017"
ANN_DIR = DATA_DIR / "annotations"
ANN_FILE = ANN_DIR / "instances_val2017.json"
CACHE_DIR = ROOT / "cache"                              # 検出キャッシュ（gitignore）
RESULTS_DIR = ROOT / "results"                          # 構造化結果（コミット対象）
MODEL_PATH = ROOT / "yolov8n.pt"                        # 既存重み（推論専用・改変しない）
# ---- モデル / 推論 ----------------------------------------------------------
MODEL_NAME = "yolov8n.pt"
INFER_CONF = 0.001     # 低conf: 全検出をキャッシュ（mAP/PR曲線のため）
OPERATING_CONF = 0.25  # 運用点（合否判定の動作点, test-design §4.2）
NMS_IOU = 0.7
IMGSZ = 640
MAX_DET = 300

# ---- 対応付け IoU -----------------------------------------------------------
IOU_PRIMARY = 0.5      # 主評価
IOU_AUX = 0.75         # 補助評価（EVAL-001 で参考記録）

# ---- 対象クラス -------------------------------------------------------------
TARGET_YOLO_IDS = [0, 1, 2, 5, 7]          # YOLO 80クラス連番 idx
TARGET_COCO_IDS = [1, 2, 3, 6, 8]          # COCO 91カテゴリ id
COCO_ID_TO_NAME = {1: "person", 2: "bicycle", 3: "car", 6: "bus", 8: "truck"}

# YOLO 80-idx -> COCO 91-cat-id 変換表（標準マッピング）
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90,
]

# ---- サブセット bin（test-design §4.3） -------------------------------------
SIZE_SMALL_MAX = 32 ** 2     # 1024 px²
SIZE_MEDIUM_MAX = 96 ** 2    # 9216 px²
SIZE_BINS = ["small", "medium", "large"]


def size_bin(area: float) -> str:
    """COCO 標準（test-design §4.3）: small ≤1024 < medium ≤9216 < large。"""
    if area <= SIZE_SMALL_MAX:
        return "small"
    if area <= SIZE_MEDIUM_MAX:
        return "medium"
    return "large"


CONGESTION_BINS = ["low", "medium", "high"]


def congestion_bin(count: int) -> str:
    """画像内の対象オブジェクト数: low 0-2 / medium 3-5 / high 6+（0個は low）。"""
    if count <= 2:
        return "low"
    if count <= 5:
        return "medium"
    return "high"


BRIGHTNESS_LOW_MAX = 80
BRIGHTNESS_MED_MAX = 160
BRIGHTNESS_BINS = ["low", "medium", "high"]


def brightness_bin(luma: float) -> str:
    """グレースケール平均輝度: low <80 / medium 80-159 / high ≥160。"""
    if luma < BRIGHTNESS_LOW_MAX:
        return "low"
    if luma < BRIGHTNESS_MED_MAX:
        return "medium"
    return "high"


MIN_SUBSET = 30   # サブセット成立の最小要件（明度・混雑=画像数, サイズ=オブジェクト数）

# ---- PR 曲線（EVAL-007） ----------------------------------------------------
PR_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95

# ---- 合格水準（criteria）。緩めない。 ---------------------------------------
OVERALL_RECALL_MIN = 0.95      # AC-1（Issue #1）
OVERALL_PRECISION_MIN = 0.70   # AC-3（Issue #1）
MEAN_IOU_MIN = 0.70            # AC-2（Issue #2）
COND_RECALL_MIN = 0.80        # AC-4a（Issue #2）
COND_PRECISION_MIN = 0.70      # AC-4b（Issue #2）
INFER_TIME_MAX_MS = 100.0      # AC-5（Issue #2, GPU 運用環境）

# summary.py が「合格水準が改変されていない」ことを検証するための期待値。
EXPECTED_CRITERIA = {
    "OVERALL_RECALL_MIN": 0.95,
    "OVERALL_PRECISION_MIN": 0.70,
    "MEAN_IOU_MIN": 0.70,
    "COND_RECALL_MIN": 0.80,
    "COND_PRECISION_MIN": 0.70,
    "INFER_TIME_MAX_MS": 100.0,
}

# ---- COCO 取得 --------------------------------------------------------------
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"
