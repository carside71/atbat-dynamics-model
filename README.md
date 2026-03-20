# atbat-dynamics-model

Statcast の投球・打席データから打席結果を予測するマルチヘッド DNN モデルです。

## 目次

- [概要](#概要)
- [プロジェクト構成](#プロジェクト構成)
- [必要条件](#必要条件)
- [セットアップ](#セットアップ)
- [設定](#設定)
- [学習](#学習)
- [テスト・性能評価](#テスト性能評価)
- [データセット](#データセット)
- [モデル](#モデル)
- [モデルグラフの可視化](#モデルグラフの可視化)
- [Linter / Formatter (Ruff)](#linter--formatter-ruff)

## 概要

投球情報（球種・球速・変化量・コース・投球軌道）と打席状況（打者・カウント・走者/アウト状況・イニング・点差）を入力として、以下の4つを同時に予測します。

| ヘッド | 出力 | タスク |
|---|---|---|
| swing_attempt | スイングしたか | 二値分類 |
| swing_result | スイング結果（foul, hit_into_play, miss） | 3クラス分類 |
| bb_type | 打球種別（ground_ball, fly_ball, line_drive, popup） | 4クラス分類 |
| regression | launch_speed, launch_angle, hit_distance_sc | 回帰 |

階層的マスク付き損失を使い、スイングしなかった場合の swing_result や、インプレーにならなかった場合の bb_type / 回帰ターゲットは損失計算から除外されます。

## プロジェクト構成

```
atbat-dynamics-model/
├── configs/                     # YAML 設定ファイル
│   ├── dnn.yaml
│   ├── dnn_change_loss_w.yaml
│   ├── dnn_mdn.yaml
│   ├── resdnn.yaml
│   ├── resdnn_cascade.yaml
│   ├── resdnn_focal.yaml
│   ├── seq_resdnn.yaml
│   └── seq_resdnn_batter_hist.yaml
├── src/
│   ├── config.py                # 設定定義 & YAML 読み込み
│   ├── train.py                 # 学習スクリプト
│   ├── test.py                  # テスト・性能評価スクリプト
│   ├── datasets/                # データセット & 前処理
│   │   ├── README.md
│   │   ├── loaders.py           #   データ読み込みユーティリティ
│   │   ├── statcast.py          #   StatcastDataset（単一投球）
│   │   ├── statcast_sequence.py #   StatcastSequenceDataset（系列対応）
│   │   └── statcast_batter_hist.py # StatcastBatterHistDataset（打者履歴対応）
│   ├── losses/                  # 損失関数
│   │   ├── focal.py             #   Focal Loss
│   │   └── multi_task.py        #   マルチタスク損失 & MDN 損失
│   ├── models/                  # モデルアーキテクチャ
│   │   ├── README.md
│   │   ├── atbat_dnn.py         #   基本マルチヘッド DNN
│   │   ├── atbat_dnn_mdn.py     #   DNN + MDN 回帰ヘッド
│   │   ├── atbat_resdnn.py      #   残差接続 + GELU
│   │   ├── atbat_resdnn_cascade.py  # カスケードヘッド付き
│   │   ├── atbat_seq_resdnn.py  #   系列エンコーダ + ResBlock
│   │   └── atbat_seq_resdnn_batter_hist.py # 打者履歴エンコーダ付き
│   └── utils/
│       ├── graph_export.py      # モデルグラフ可視化
│       ├── logging.py           # ログ出力
│       └── model_io.py          # モデル構築・保存・復元
├── notes/                       # データ構築・分析ノートブック
│   ├── 00_build_dataset/        #   前処理パイプライン
│   ├── 01_analysis/             #   データ分析
│   └── locals/                  #   ローカル実行用スクリプト
├── scripts/
│   ├── add_game_info_and_rebuild.py  # game_pk/game_date 付与 & 時系列分割 & 打者履歴構築
│   ├── export_model_graph.py    # モデルグラフ構造の画像出力
│   ├── run_container_mac.sh     # Mac 用コンテナ起動
│   └── run_container_wsl.sh     # WSL 用コンテナ起動
├── Dockerfile
├── pyproject.toml
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
  label_smoothing: 0.0   # > 0 で Label Smoothing 有効（0.1 程度が一般的）
```

設定可能な全フィールドは `src/config.py` および `configs/` 内の各 YAML ファイルを参照してください。

## 学習

```bash
# YAML 設定ファイルを指定して実行
python3 src/train.py --config configs/resdnn.yaml

# デフォルト設定で実行（YAML 不要）
python3 src/train.py
```

学習が完了すると、`output_dir`（デフォルト: `/workspace/outputs/atbat-dynamics-model/`）に以下が保存されます。

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
python3 src/test.py --config configs/resdnn.yaml --split test

# 検証データで評価
python3 src/test.py --config configs/resdnn.yaml --split val

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

## データセット

`/workspace/datasets/statcast-customized/` に配置された Statcast データを使用します。

| ディレクトリ | 内容 |
|---|---|
| `data/` | 年月別の Parquet ファイル（例: `statcast_2024_04.parquet`） |
| `stats/` | カテゴリカル特徴量のラベル対応テーブル（CSV） |
| `split/` | 学習/検証/テスト分割の `at_bat_id` リスト（CSV）※時系列分割 |
| `batter_history/` | 打者履歴ルックアップテーブル（Parquet） |

データ分割は **時系列分割**（`game_date` ベース）で行われます。学習データは 2024-06-30 以前、検証データは 2024-07-01〜2024-10-30、テストデータは 2025-03-15 以降です。ダブルヘッダーを正しく区別するため `game_pk`（試合ごとに一意な ID）を使用します。

前処理は `notes/00_build_dataset/` 配下のノートブックで段階的に実行します。打者履歴テーブルの構築は `scripts/add_game_info_and_rebuild.py` で行います。

データセットクラスや読み込みユーティリティの詳細は [src/datasets/README.md](src/datasets/README.md) を参照してください。

## モデル

すべてのモデルは **埋め込み → バックボーン → マルチヘッド** の3段構成で、レジストリパターンで管理されています。

| `architecture` | 設定ファイル例 | 説明 |
|---|---|---|
| `atbat_dnn` | `dnn.yaml` | 基本マルチヘッド DNN (ReLU + BatchNorm) |
| `atbat_dnn_mdn` | `dnn_mdn.yaml` | 回帰ヘッドを MDN に置換 |
| `atbat_resdnn` | `resdnn.yaml` | 残差接続 + GELU + LayerNorm |
| `atbat_resdnn_cascade` | `resdnn_cascade.yaml` | 上記 + カスケードヘッド（ヘッド間情報伝達） |
| `atbat_seq_resdnn` | `seq_resdnn.yaml` | 打席内系列エンコーダ (GRU/Transformer) + ResBlock |
| `atbat_seq_resdnn_batter_hist` | `seq_resdnn_batter_hist.yaml` | 上記 + 階層 GRU 打者履歴エンコーダ |

各モデルのアーキテクチャ詳細・図解・追加方法は [src/models/README.md](src/models/README.md) を参照してください。

## モデルグラフの可視化

登録済みモデルの計算グラフ構造を画像ファイル（PNG / PDF / SVG）として出力できます。

```bash
# 全モデルを一括出力
python3 scripts/export_model_graph.py --all

# YAML 設定ファイルから単一モデルを出力
python3 scripts/export_model_graph.py --config configs/resdnn.yaml

# アーキテクチャ名を直接指定
python3 scripts/export_model_graph.py --arch atbat_dnn_mdn
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--config` | — | YAML 設定ファイルパス（排他） |
| `--arch` | — | アーキテクチャ名（排他） |
| `--all` | — | 全登録モデルを出力（排他） |
| `--output-dir` | `outputs/graphs` | 保存先ディレクトリ |
| `--format` | `png` | 出力形式（`png` / `pdf` / `svg`） |
| `--backend` | `torchview` | 描画バックエンド（`torchview` / `torchviz`） |
| `--depth` | `3` | モジュール展開深度（torchview のみ） |
| `--batch-size` | `2` | ダミー入力のバッチサイズ |

### バックエンド

- **torchview**（推奨）: モジュール階層を構造的に描画。`--depth` で展開の深さを制御可能
- **torchviz**: autograd の計算グラフを描画。各演算ノードが詳細に表示される

### 依存パッケージ

- Python: `torchview`, `torchviz`, `graphviz`（`requirements.txt` に記載済み）
- システム: `graphviz`（`apt-get install graphviz`、Dockerfile に記載済み）

## Linter / Formatter (Ruff)

コード品質を統一するために [Ruff](https://docs.astral.sh/ruff/) を導入しています。ルールは `pyproject.toml` に定義されています。

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
