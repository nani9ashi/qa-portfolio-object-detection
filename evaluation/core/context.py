"""全ケースが共有する評価コンテキスト（キャッシュ＋GT索引＋設定）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class EvalContext:
    coco: object                          # pycocotools COCO（EVAL-002 用）
    img_ids: List[int]                    # 評価対象の画像 id（決定的に sorted[:N]）
    img_paths: Dict[int, Path]            # {iid: 画像パス}
    detections: Dict[int, List[Dict]]     # {iid: [{"score","box","cls","area"}]}（低conf全量）
    gt_by_img_cls: Dict                   # {(iid, cls): [gt, ...]}
    congestion_count: Dict[int, int]      # {iid: 対象GT数}
    img_meta: Dict[int, Dict]             # {iid: {"mean_luma","congestion_count"}}
    device: str
    op_records: List[Dict] = field(default_factory=list)  # 運用点(conf0.25,IoU0.5)の記録

    @property
    def num_images_evaluated(self) -> int:
        return len(self.img_ids)
