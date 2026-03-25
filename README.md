# atbat-dynamics-model

Statcast のデータを用いて未来の打席結果を予測する AI (Deep Neural Network, DNN) モデルを構築してみた。

## 目次

- [デモ](#デモ)
- [概要](#概要)
- [プロジェクト構成](#プロジェクト構成)
- [必要条件](#必要条件)
- [セットアップ](#セットアップ)
- [設定](#設定)
- [学習](#学習)
- [テスト・性能評価](#テスト性能評価)
- [データセット](#データセット)
- [モデル](#モデル)
- [ツール](#ツール)

## デモ

構築した DNN モデルを用いて2025シーズンの大谷の全打席結果を予測してみた。

<a href="https://carside71.github.io/atbat-dynamics-model/viewer_ohtani.html">
  <img src="figs/viewer_preview.svg" alt="Prediction Viewer デモ" width="800">
</a>

**[▶ ライブデモを開く](https://carside71.github.io/atbat-dynamics-model/viewer_ohtani.html)**

---

## 概要

### 予測ターゲット

投球情報（球種・球速・変化量・コース・投球軌道）と打席状況（打者・カウント・走者/アウト状況・イニング・点差）を入力として、以下の4つを同時に予測します。

| ヘッド | 出力 | タスク |
|---|---|---|
| swing_attempt | スイングしたか | 二値分類 |
| swing_result | スイング結果（foul, hit_into_play, miss） | 3クラス分類 |
| bb_type | 打球種別（ground_ball, fly_ball, line_drive, popup） | 4クラス分類 |
| regression | launch_speed, launch_angle, hit_distance_sc, spray_angle | 回帰 |

階層的マスク付き損失を使い、スイングしなかった場合の swing_result や、インプレーにならなかった場合の bb_type / 回帰ターゲットは損失計算から除外されます。

### モデルスコープ

`model_scope` 設定により、予測タスクの範囲を切り替えて **分離学習** が可能です。

| `model_scope` | 予測対象 | 学習データ | 用途 |
|---|---|---|---|
| `all`（デフォルト） | 全4タスク | 全サンプル | 統合モデル（従来動作） |
| `swing_attempt` | swing_attempt のみ | 全サンプル | スイング判定の専用モデル |
| `outcome` | swing_result + bb_type + regression | swing_attempt=1 のみ | スイング後の結果予測専用モデル |
| `classification` | swing_attempt + swing_result + bb_type | 全サンプル | 3分類タスクのみ（回帰なし） |
| `regression` | regression のみ | swing_attempt=1 のみ | 回帰予測専用モデル |

タスクごとに分離学習することで、各タスクに最適化された専用モデルを構築できます。

## プロジェクト構成

```
atbat-dynamics-model/
├── configs/                     # YAML 設定ファイル（model_scope 別にディレクトリ分割）
│   ├── all/                     #   全タスク統合モデル
│   ├── swing_attempt/           #   swing_attempt 専用モデル
│   ├── outcome/                 #   outcome 専用モデル（SR + BT + Reg）
│   ├── classification/          #   classification 専用モデル（SA + SR + BT）
│   └── regression/              #   regression 専用モデル
├── docs/                        # GitHub Pages デモサイト
├── figs/                        # ドキュメント用画像
├── src/
│   ├── config.py                # 設定定義 & YAML 読み込み
│   ├── train.py                 # 学習スクリプト
│   ├── test.py                  # テスト・性能評価スクリプト
│   ├── datasets/                # データセット & 前処理
│   ├── losses/                  # 損失関数
│   ├── models/                  # モデルアーキテクチャ
│   └── utils/                   # ユーティリティ関数群
├── tests/                       # テストコード
├── notes/                       # データ構築・分析ノートブック
│   ├── 00_build_dataset/        #   前処理パイプライン
│   └── 01_analysis/             #   データ分析
├── scripts/
│   ├── run_container_mac.sh     # Mac 用コンテナ起動
│   └── run_container_wsl.sh     # WSL 用コンテナ起動
├── tools/
│   ├── build_dataset/           # データセット構築パイプライン
│   ├── export_graph/            # モデルグラフ構造の画像出力
│   ├── generate_viewer/         # 予測ビューア HTML 生成
│   └── plot_curves/             # 学習曲線プロット
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
  dataset_dir: /workspace/datasets/statcast-customized-v2

model:
  model_scope: all               # "all" | "swing_attempt" | "outcome" | "classification" | "regression"
  backbone_type: resdnn           # "dnn" | "resdnn"
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
  loss_weight_physics: 0.0      # > 0 で物理的整合性損失を有効化（0.001〜0.01 推奨）
  physics_margin_degrees: 2.0   # 境界付近のマージン（度）
```

設定可能な全フィールドは `src/config.py` および `configs/` 内の各 YAML ファイルを参照してください。

## 学習

```bash
# YAML 設定ファイルを指定して実行（全タスク統合モデル）
python3 src/train.py --config configs/all/dnn.yaml

# swing_attempt 専用モデルの学習
python3 src/train.py --config configs/swing_attempt/dnn.yaml

# outcome 専用モデルの学習（swing_attempt=1 のサンプルのみ使用）
python3 src/train.py --config configs/outcome/dnn.yaml

# classification 専用モデルの学習（SA + SR + BT、回帰なし）
python3 src/train.py --config configs/classification/dnn.yaml

# regression 専用モデルの学習（swing_attempt=1 のサンプルのみ使用）
python3 src/train.py --config configs/regression/dnn.yaml

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
python3 src/test.py --config configs/all/resdnn.yaml --split test

# 検証データで評価
python3 src/test.py --config configs/all/resdnn.yaml --split val

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

### 予測値の保存

`--save-predictions` フラグを付けると、サンプルごとの予測値・GT・入力特徴量を NPZ + メタデータ JSON として保存します。後述の [Prediction Viewer](#予測結果の可視化prediction-viewer) で可視化に使用します。

```bash
python3 src/test.py --config configs/all/resdnn.yaml --save-predictions
```

| ファイル | 内容 |
|---|---|
| `predictions_{split}.npz` | 予測確率・logits・GT・入力特徴量（NumPy 圧縮アーカイブ） |
| `predictions_meta_{split}.json` | ラベル名・正規化パラメータ・カテゴリラベルマッピング |

## データセット

`/workspace/datasets/statcast-customized-v2/` にフラット構造で配置されたデータセットを使用します。

| ファイル | 内容 |
|---|---|
| `pitches.parquet` | 全投球データ（単一ファイル） |
| `stats_*.csv` | カテゴリカル特徴量のラベル対応テーブル |
| `train_at_bat_ids.csv` / `valid_at_bat_ids.csv` / `test_at_bat_ids.csv` | 時系列分割の `at_bat_id` リスト |
| `batter_game_history.parquet` / `atbat_row_indices.parquet` | 打者履歴ルックアップテーブル |
| `atbat_metadata.parquet` / `player_names.json` | ビューア用メタデータ |

データ分割は **時系列分割**（`game_date` ベース）で行われます。学習データは 2024-06-30 以前、検証データは 2024-07-01〜2024-10-30、テストデータは 2025-03-15 以降です。ダブルヘッダーを正しく区別するため `game_pk`（試合ごとに一意な ID）を使用します。

`tools/build_dataset/` パッケージにモジュール化されたパイプラインで構築します。詳細は [tools/README.md](tools/README.md) を参照してください。

データセットクラスや読み込みユーティリティの詳細は [src/datasets/README.md](src/datasets/README.md) を参照してください。

## モデル

すべてのモデルは `ComposableModel` が **埋め込み → バックボーン → マルチヘッド** の各コンポーネントを YAML 設定に基づいて組み立てる構成で、レジストリパターン (`utils/registry.py`) で管理されています。

| 設定ファイル例 | scope | 説明 |
|---|---|---|
| `all/dnn.yaml` | all | 基本マルチヘッド DNN (ReLU + BatchNorm) |
| `all/dnn_mdn.yaml` | all | 回帰ヘッドを MDN に置換 |
| `all/resdnn.yaml` | all | 残差接続 + GELU + LayerNorm |
| `all/resdnn_cascade.yaml` | all | ResBlock + カスケードヘッド（ヘッド間情報伝達） |
| `all/resdnn_cascade_physics.yaml` | all | ResBlock + カスケード + 物理的整合性損失 |
| `all/resdnn_focal.yaml` | all | ResBlock + Focal Loss |
| `all/resdnn_pitch_seq.yaml` | all | 打席内系列エンコーダ (GRU/Transformer) + ResBlock |
| `all/resdnn_pitch_seq_batter_hist.yaml` | all | 上記 + 階層 GRU 打者履歴エンコーダ |
| `swing_attempt/dnn.yaml` | swing_attempt | swing_attempt 専用モデル |
| `outcome/dnn.yaml` | outcome | outcome 専用モデル（SA=1 サンプルのみ） |
| `classification/dnn.yaml` | classification | 3分類タスク専用モデル（回帰なし） |
| `regression/dnn.yaml` | regression | 回帰専用モデル（SA=1 サンプルのみ） |

各モデルのアーキテクチャ詳細・図解・追加方法は [src/models/README.md](src/models/README.md) を参照してください。

## ツール

`tools/` ディレクトリには、データセット構築・モデル可視化・学習分析などの各種ユーティリティが含まれています。すべて `python -m tools.<ツール名>` で実行できます。

| ツール | 説明 |
|---|---|
| `build_dataset` | Statcast データからモデル学習用データセットを構築するパイプライン |
| `export_graph` | モデルの計算グラフ構造を画像（PNG / PDF / SVG）として出力 |
| `generate_viewer` | テスト結果をインタラクティブに閲覧できる自己完結型 HTML ビューアを生成 |
| `plot_curves` | `train.py` の学習履歴（`history.json`）から学習曲線をプロット |

各ツールの詳細な使い方・オプションは [tools/README.md](tools/README.md) を参照してください。
