"""画像のグレースケール平均輝度（0-255）。EVAL-006 の明度サブセット分類に用いる。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2


def mean_luminance(img_path: Path) -> Optional[float]:
    """BGR 画像をグレースケール化し、全画素の平均輝度を返す。読めなければ None。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())
