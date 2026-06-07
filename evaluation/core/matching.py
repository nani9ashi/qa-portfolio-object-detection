"""検出と正解の貪欲 IoU マッチング（(画像,クラス)単位）。

マッチングのアルゴリズム: 検出を信頼度降順に見て、各検出に「未マッチGTの最大IoU」を
当て、IoU≥閾値なら TP（そのGTを消費）、未達なら FP。残った未マッチGTは FN。
各記録に検出box/GTの面積とサイズbinを持たせ、EVAL-004 のサイズ帰属を可能にする。
"""
from __future__ import annotations

from typing import Dict, List

from evaluation import config


def iou_xyxy(a, b) -> float:
    """xyxy 形式の2box の IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image_class(dets: List[Dict], gts: List[Dict], iou_thr: float) -> List[Dict]:
    """1画像・1クラス分の検出とGTをマッチし、TP/FP/FN 記録のリストを返す。

    dets: [{"score","box":[x1,y1,x2,y2],"area"}]（conf フィルタ済みを想定）
    gts : [{"box":[x1,y1,x2,y2],"area","size_bin"}]
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    matched = [False] * len(gts)
    records: List[Dict] = []
    for i in order:
        d = dets[i]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = iou_xyxy(d["box"], g["box"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched[best_j] = True
            g = gts[best_j]
            records.append({
                "type": "TP", "iou": best_iou,
                "det_box": d["box"], "gt_box": g["box"],
                "det_area": d["area"], "det_size": config.size_bin(d["area"]),
                "gt_area": g["area"], "gt_size": g["size_bin"],
            })
        else:
            records.append({
                "type": "FP",
                "det_box": d["box"],
                "det_area": d["area"], "det_size": config.size_bin(d["area"]),
            })
    for j, g in enumerate(gts):
        if not matched[j]:
            records.append({"type": "FN", "gt_box": g["box"],
                            "gt_area": g["area"], "gt_size": g["size_bin"]})
    return records


def build_records(detections, gt_by_img_cls, img_ids, conf_thr, iou_thr) -> List[Dict]:
    """全画像・全対象クラスでマッチングし、(image_id, cls, rec) のタグ付きリストを返す。

    detections   : {iid: [{"score","box","cls","area"}, ...]}（低conf全量キャッシュ）
    gt_by_img_cls: {(iid, cls): [gt, ...]}
    戻り値        : [{"image_id","cls","rec": <record>}, ...]
    """
    out: List[Dict] = []
    for iid in img_ids:
        dets_all = detections.get(iid, [])
        for cid in config.TARGET_COCO_IDS:
            dets = [d for d in dets_all if d["cls"] == cid and d["score"] >= conf_thr]
            gts = gt_by_img_cls.get((iid, cid), [])
            if not dets and not gts:
                continue
            for rec in match_image_class(dets, gts, iou_thr):
                out.append({"image_id": iid, "cls": cid, "rec": rec})
    return out
