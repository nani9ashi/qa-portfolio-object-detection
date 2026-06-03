"""正解アノテーションの読み込みと索引化。iscrowd は除外する（曖昧な群アノテのため）。"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from evaluation import config


def load_gt(coco, img_ids) -> Tuple[Dict[Tuple[int, int], List[Dict]], Dict[int, int]]:
    """正解を (画像, クラス) で索引化し、画像ごとの混雑度カウントも返す。

    戻り値:
      gt_by_img_cls : {(iid, coco_cat_id): [{"box":[x1,y1,x2,y2], "area", "size_bin"}]}
      congestion_count : {iid: 対象クラスGTの個数（iscrowd除外）}
    面積は COCO の area フィールド（セグメンテーション面積。pycocotools/COCOeval と整合）。
    """
    gt_by_img_cls: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    congestion_count: Dict[int, int] = {}
    for iid in img_ids:
        ann_ids = coco.getAnnIds(imgIds=int(iid), catIds=config.TARGET_COCO_IDS, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            x, y, w, h = a["bbox"]
            area = float(a["area"])
            gt_by_img_cls[(int(iid), int(a["category_id"]))].append({
                "box": [float(x), float(y), float(x + w), float(y + h)],
                "area": area,
                "size_bin": config.size_bin(area),
            })
        congestion_count[int(iid)] = len(anns)
    return dict(gt_by_img_cls), congestion_count
