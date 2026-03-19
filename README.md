# atbat-dynamics-model

Statcast の投球・打席データから打席結果を予測するマルチヘッド DNN モデルです。

## 目次

- [概要](#概要)
- [プロジェクト構成](#プロジェクト構成)
- [必要条件](#必要条件)
- [セットアップ](#セットアップ)
- [データセット](#データセット)
- [設定](#設定)
- [学習](#学習)
- [テスト・性能評価](#テスト性能評価)
- [Linter / Formatter (Ruff)](#linter--formatter-ruff)

## 概要

投球情報（球種・球速・変化量・コース）と打席状況（打者・カウント・走者/アウト状況・イニング・点差）を入力として、以下の4つを同時に予測します。

| ヘッド | 出力 | タスク |
|---|---|---|
| swing_attempt | スイングしたか | 二値分類 |
| swing_result | スイング結果（foul, hit_into_play, swinging_strike 等） | 9クラス分類 |
| bb_type | 打球種別（ground_ball, fly_ball, line_drive, popup） | 4クラス分類 |
| regression | launch_speed, launch_angle, hit_distance_sc | 回帰 |

階層的マスク付き損失を使い、スイングしなかった場合の swing_result や、インプレーにならなかった場合の bb_type / 回帰ターゲットは損失計算から除外されます。

## プロジェクト構成

```
atbat-dynamics-model/
├── configs/
│   ├── dnn.yaml               # 基本 DNN
│   ├── dnn_w_mdn.yaml         # DNN + MDN 回帰ヘッド
│   ├── res_dnn.yaml           # 残差接続 + GELU
│   ├── res_dnn_focal.yaml     # 残差接続 + Focal Loss
│   └── resdnn_cascade.yaml    # 残差接続 + カスケードヘッド
├── src/
│   ├── config.py              # 設定定義 & YAML読み込み
│   ├── dataset.py             # データセット & 前処理
│   ├── train.py               # 学習スクリプト
│   ├── test.py                # テスト・性能評価スクリプト
│   ├── losses/                # 損失関数
│   │   ├── focal.py             #   Focal Loss
│   │   └── multi_task.py        #   マルチタスク損失 & MDN損失
│   ├── models/                # モデルアーキテクチャ
│   │   ├── atbat_dnn.py         #   基本マルチヘッド DNN
│   │   ├── atbat_dnn_mdn.py     #   DNN + MDN 回帰ヘッド
│   │   ├── atbat_resdnn.py      #   残差接続 + GELU
│   │   └── atbat_resdnn_cascade.py  # カスケードヘッド付き
│   └── utils/
│       ├── logging.py           # ログ出力
│       └── model_io.py          # モデル構築・保存・復元
├── scripts/
│   ├── run_container_mac.sh   # Mac 用コンテナ起動
│   └── run_container_wsl.sh   # WSL 用コンテナ起動
├── Dockerfile
└── requirements.txt
```

## 必要条件

- Python 3.10+
- PyTorch 2.x（CUDA 対応推奨）
- 主な依存: `numpy`, `pandas`, `pyyaml`, `tqdm`, `scikit-learn`

## セットアップ

### ローカル環境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker

```bash
docker build -t atbat-dynamics-model-image:latest .
./scripts/run_container_wsl.sh   # WSL の場合
./scripts/run_container_mac.sh   # Mac の場合
```

スクリプト内の `SRC_DIR`, `DATA_DIR`, `OUT_DIR` を環境に合わせて編集してください。

## データセット

`/workspace/datasets/statcast-customized/` に配置された Statcast データを使用します。

- `data/` — 年月別の Parquet ファイル（例: `statcast_2024_04.parquet`）
- `stats/` — カテゴリカル特徴量のラベル対応テーブル（CSV）
- `split/` — 学習/検証/テスト分割の `at_bat_id` リスト（CSV）

データ分割は `split/` ディレクトリの `at_bat_id` により行われます。各 Parquet ファイルに含まれる `at_bat_id` カラムと以下の CSV ファイルを照合して分割します。

| ファイル | 用途 |
|---|---|
| `split/train_at_bat_ids.csv` | 学習データ |
| `split/valid_at_bat_ids.csv` | 検証データ |
| `split/test_at_bat_ids.csv` | テストデータ |

## 設定

YAML ファイル（`configs/` ディレクトリ）で `data`, `model`, `train` の3セクションを設定できます。
各フィールドは省略可能で、省略時はデフォルト値が使われます。

```yaml
data:
  data_dir: /workspace/datasets/statcast-customized/data
  stats_dir: /workspace/datasets/statcast-customized/stats
  split_dir: /workspace/datasets/statcast-customized/split

model:
  architecture: atbat_resdnn     # モデル名
  backbone_hidden: [512, 512, 256, 256, 128]
  head_hidden: [64]
  dropout: 0.2

train:
  batch_size: 4096
  num_epochs: 30
  lr: 1.0e-3
  device: cuda
  focal_gamma: 0.0       # > 0 で Focal Loss 有効
  use_class_weight: false # true でクラス頻度の逆数重み付け
```

設定可能な全フィールドは `configs/` 内の各 YAML ファイルを参照してください。

### 利用可能なモデル

| `architecture` | 設定ファイル例 | 説明 |
|---|---|---|
| `atbat_dnn` | `dnn.yaml` | 基本マルチヘッド DNN (ReLU + BatchNorm) |
| `atbat_dnn_mdn` | `dnn_w_mdn.yaml` | 回帰ヘッドを MDN に置換 |
| `atbat_resdnn` | `res_dnn.yaml` | 残差接続 + GELU + LayerNorm |
| `atbat_resdnn_cascade` | `resdnn_cascade.yaml` | 上記 + カスケードヘッド（ヘッド間情報伝達） |

## 学習

```bash
# YAML 設定ファイルを指定して実行
python3 src/train.py --config configs/res_dnn.yaml

# デフォルト設定で実行（YAML 不要）
python3 src/train.py
```

学習が完了すると、`output_dir`（デフォルト: `/workspace/outputs/`）に以下が保存されます。

| ファイル | 内容 |
|---|---|
| `best_model.pt` | 検証損失が最小のモデル重み |
| `final_model.pt` | 最終エポックのモデル重み |
| `history.json` | エポックごとの損失・精度の履歴 |
| `norm_params.json` | 入力・ターゲットの正規化パラメータ |
| `model_config.json` | モデル構造の設定 |

## テスト・性能評価

学習済みモデルの性能をテストデータまたは検証データで評価できます。

```bash
# テストデータで評価
python3 src/test.py --config configs/res_dnn.yaml --split test

# 検証データで評価
python3 src/test.py --config configs/res_dnn.yaml --split val

# モデルディレクトリ・ファイルを明示的に指定
python3 src/test.py --model-dir /path/to/model --model-file best_model.pt
```

評価されるメトリクス:

| タスク | メトリクス |
|---|---|
| swing_attempt | Accuracy, F1, ROC AUC, Confusion Matrix |
| swing_result | Accuracy, F1 (macro/weighted), Per-class Report |
| bb_type | Accuracy, F1 (macro/weighted), Per-class Report |
| regression | MAE, RMSE, R²（元スケールに逆変換） |

結果はコンソールに表示されるほか、`test_results_test.json` / `test_results_val.json` としてモデルディレクトリに保存されます。

## Linter / Formatter (Ruff)

コード品質を統一するために [Ruff](https://docs.astral.sh/ruff/) を導入しています。Ruff は linter と formatter の両方を兼ねた高速な Python ツールです。

### 設定

ルールは `pyproject.toml` に定義されています。

| 項目 | 値 |
|---|---|
| 行長制限 | 120 文字 |
| Lint ルール | pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear, flake8-simplify |
| フォーマット | ダブルクォート、スペースインデント |

### CLI での使い方

```bash
# Lint チェック
ruff check src/

# Lint チェック + 自動修正
ruff check --fix src/

# フォーマット
ruff format src/

# フォーマット差分確認（変更なし）
ruff format --check src/
```

### VS Code での自動フォーマット

Docker コンテナ起動スクリプト（`run_container_wsl.sh` / `run_container_mac.sh`）に `devcontainer.metadata` ラベルが設定されています。VS Code の **Dev Containers** 拡張（`ms-vscode-remote.remote-containers`）でコンテナにアタッチすると、以下が自動的に行われます。

1. **Ruff 拡張機能**（`charliermarsh.ruff`）が自動インストールされる
2. **保存時（Ctrl+S）** にフォーマット・lint 自動修正・import ソートが実行される

> **前提**: ホスト側の VS Code に [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 拡張がインストールされている必要があります。
