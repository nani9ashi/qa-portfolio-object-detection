"""COCO val2017 の取得（必要な画像とアノテーションをダウンロードする）。

冪等: 既にあるファイルは取得しない。落ちた画像はリトライし、最終的に取得できた画像のみ評価対象。
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from evaluation import config


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "qa-eval/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def ensure_annotations(log=print) -> None:
    if config.ANN_FILE.exists():
        log(f"[data] アノテーション既存: {config.ANN_FILE}")
        return
    config.ANN_DIR.mkdir(parents=True, exist_ok=True)
    log("[data] COCO アノテーションをダウンロード（約241MB, 初回のみ）…")
    zip_path = config.DATA_DIR / "annotations_trainval2017.zip"
    if not zip_path.exists():
        _download(config.COCO_ANN_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("annotations/instances_val2017.json", path=str(config.DATA_DIR))
    log(f"[data] 展開完了: {config.ANN_FILE}")


def ensure_images(coco, img_ids, log=print) -> Dict[int, Path]:
    """各画像を data/val2017 に用意（無ければ並列DL）。取得済みの {iid: path} を返す。"""
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, Path] = {}
    to_download: List[Tuple[str, Path]] = []
    for iid in img_ids:
        info = coco.loadImgs(int(iid))[0]
        fname = info["file_name"]
        dest = config.IMG_DIR / fname
        paths[int(iid)] = dest
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = info.get("coco_url") or config.COCO_IMG_URL.format(file_name=fname)
        to_download.append((url, dest))
    if to_download:
        log(f"[data] 画像 {len(to_download)} 枚をダウンロード（並列）…")

        def fetch(job: Tuple[str, Path]) -> bool:
            url, dest = job
            for _ in range(3):
                try:
                    _download(url, dest, timeout=60)
                    return True
                except Exception:
                    continue
            log(f"  [warn] DL失敗: {dest.name}")
            return False

        with ThreadPoolExecutor(max_workers=16) as ex:
            ok = sum(1 for r in ex.map(fetch, to_download) if r)
        log(f"[data] DL完了 {ok}/{len(to_download)}")
    return {iid: p for iid, p in paths.items() if p.exists() and p.stat().st_size > 0}


def prepare(num_images: int, log=print):
    """アノテーション＋画像を用意し、(coco, 評価対象img_ids, {iid:path}) を返す。"""
    ensure_annotations(log)
    from pycocotools.coco import COCO
    coco = COCO(str(config.ANN_FILE))
    all_ids = sorted(coco.getImgIds())
    img_ids = all_ids[:num_images]
    paths = ensure_images(coco, img_ids, log)
    avail = [i for i in img_ids if i in paths]
    log(f"[data] 評価対象: {len(avail)}/{len(img_ids)} 枚（val2017 全 {len(all_ids)} 枚中）")
    return coco, avail, paths
