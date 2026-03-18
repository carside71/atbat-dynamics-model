# atbat-dynamics-model

Statcast の投球・打席データから打席結果を予測するマルチヘッド DNN モデルです。

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
│   └── config.yaml          # 学習設定（YAML）
├── src/
│   ├── config.py             # 設定定義 & YAML読み込み
│   ├── dataset.py            # データセット & 前処理
│   ├── model.py              # マルチヘッド DNN モデル
│   ├── train.py              # 学習スクリプト
│   └── test.py               # テスト・性能評価スクリプト
├── scripts/
│   ├── run_container_mac.sh  # Mac 用コンテナ起動
│   └── run_container_wsl.sh  # WSL 用コンテナ起動
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

## 設定

YAML ファイル（`configs/config.yaml`）で `data`, `model`, `train` の3セクションを設定できます。
各フィールドは省略可能で、省略時はデフォルト値が使われます。

```yaml
data:
  train_years: [2017, 2018, 2019, 2020, 2021, 2022, 2023]
  val_years: [2024]
  test_years: [2025]

model:
  backbone_hidden: [512, 256, 128]
  head_hidden: [64]
  dropout: 0.2

train:
  batch_size: 4096
  num_epochs: 30
  lr: 1.0e-3
  device: cuda
```

設定可能な全フィールドは [configs/config.yaml](configs/config.yaml) を参照してください。

## 学習

```bash
# YAML 設定ファイルを指定して実行
python3 src/train.py --config configs/config.yaml

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
# テストデータ（2025年）で評価
python3 src/test.py --config configs/config.yaml --split test

# 検証データ（2024年）で評価
python3 src/test.py --config configs/config.yaml --split val

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
