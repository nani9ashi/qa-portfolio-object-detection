# SETUP — 環境構築と再現手順

本評価を手元で再現するための手順をまとめる。本番運用相当の **GPU 環境（CUDA 対応 PyTorch）** を前提とする。AC-5（推論時間）は GPU で評価するため、CPU では推論時間の判定が再現しない。

## 1. 評価本体の実行 (Windows / PowerShell)

```powershell
# 1. 仮想環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. 依存（PyTorch は CUDA 版を CUDA インデックスから先に入れる）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics pycocotools opencv-python polars

# 3. 評価の実行（COCO val2017 を取得 → 推論 → EVAL-001〜008 → 総合判定）
python evaluation\run_eval.py --num-images 5000 --device cuda
```

実行すると `results/` に各 EVAL ケースの JSON、PR 曲線 CSV、総合判定 `summary.json` が出力される。

- COCO データ・モデル重み・検出キャッシュはリポジトリに含めない（`.gitignore`）。
- `requirements.txt` は実行環境の厳密なピンを記録している（CUDA 版 torch を含むため、再現時は上記のとおり CUDA インデックスからの導入が必要）。
- YOLO の推論は決定的なため、同一データ・同一モデル・同一運用点（conf 0.25 / IoU 0.5）であれば 1 回の実行で結果が一致する。

## 2. README 用の図の再生成

`results/` が揃った状態で、評価成果物（JSON・PR 曲線 CSV）から README 用の図を再生成する。数値も基準線も results 由来で、図と本文・判定の数値が乖離しない。

```powershell
pip install matplotlib japanize-matplotlib
python make_figures.py --results results --out docs/images
```

生成物: `docs/images/01_overall_pr_curve.png` / `02_recall_precision_by_condition.png` / `03_miss_breakdown_by_size.png`。

## 3. 合格基準の整合性チェック（GPU 不要・数秒）

合格水準が後から緩められていないか、合否ゲートに統合スコア（F1 / mAP）が混入していないかを検証する。CI（`.github/workflows/criteria-integrity.yml`）でも毎 push / PR で自動実行される。

```powershell
pip install pytest
pytest -q evaluation\test_criteria_integrity.py
```

> このチェックが `results/summary.json` を読むため、`summary.json` は追跡対象（コミット済み）である必要がある。サイズが小さく、評価結果そのものの証跡でもあるため、リポジトリに含める運用とする。
