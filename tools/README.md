# Tools

各種ユーティリティツールのドキュメントです。すべてのツールは `python -m tools.<ツール名>` で実行できます。

## 目次

- [build\_dataset — データセット構築パイプライン](#build_dataset--データセット構築パイプライン)
- [export\_graph — モデルグラフの可視化](#export_graph--モデルグラフの可視化)
- [generate\_viewer — 予測結果の可視化（Prediction Viewer）](#generate_viewer--予測結果の可視化prediction-viewer)
- [plot\_curves — 学習曲線プロット](#plot_curves--学習曲線プロット)

---

## build\_dataset — データセット構築パイプライン

`tools/build_dataset/` パッケージにモジュール化されたパイプラインで、Statcast データからモデル学習用データセットを構築します。実行は `notes/00_build_dataset/build_dataset.ipynb` を開いて全セル実行するか、Python から直接呼び出します。

```python
from tools.build_dataset import run_pipeline
run_pipeline()
```

| Step | 内容 |
|------|------|
| 1. Filter | 2000球以上の打者を抽出 |
| 2. Features | カラム選択・ゲームステート・軌道特徴量・正規化 |
| 3. Labels | description解析・カテゴリカルエンコード・stats生成 |
| 4. Splits | 時系列分割・打者履歴構築・保存 |
| 5. Quality | ソースデータの品質レポート |

中間ファイルは生成せず、全処理をインメモリで実行します。各ステップの処理結果はノートブック上で視覚的に確認でき、実行後も見直せます。

---

## export\_graph — モデルグラフの可視化

登録済みモデルの計算グラフ構造を画像ファイル（PNG / PDF / SVG）として出力できます。

```bash
# 全モデルを一括出力
python3 -m tools.export_graph --all

# YAML 設定ファイルから単一モデルを出力
python3 -m tools.export_graph --config configs/all/resdnn.yaml

# アーキテクチャ名を直接指定
python3 -m tools.export_graph --arch atbat_dnn_mdn
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

---

## generate\_viewer — 予測結果の可視化（Prediction Viewer）

テスト結果をサンプル単位でインタラクティブに閲覧できる自己完結型 HTML ビューアを生成します。

### 前準備: 選手名メタデータの構築

ビューアで選手名を表示するには、事前に `player_names.json` を構築しておく必要があります。`atbat_metadata.parquet` はデータセット構築パイプライン（Step 4）で自動生成されます。**初回のみ実行すれば OK です**（データセットが変わらない限り再実行不要）。

```bash
python3 -m tools.generate_viewer.metadata
```

処理内容:
1. `atbat_metadata.parquet` から打者・投手の MLBAM ID を収集
2. 元の Statcast CSV から投手名を収集
3. MLB Stats API から不足分の選手名を一括取得
4. `player_names.json` を出力

| オプション | デフォルト | 説明 |
|---|---|---|
| `--dataset-dir` | `/workspace/datasets/statcast-customized-v2` | データセットディレクトリ |
| `--raw-csv-dir` | `/workspace/datasets/statcast` | 元の Statcast CSV ディレクトリ（省略可） |

### 使い方

```bash
# 1. メタデータ構築（初回のみ）
python3 -m tools.generate_viewer.metadata

# 2. テスト実行時に予測データを保存
python3 src/test.py --config configs/all/resdnn_pitch_seq_batter_hist.yaml --save-predictions

# 3. ビューア HTML を生成
python3 -m tools.generate_viewer \
  --pred-dir outputs/.../test/2026-03-20-120000 \
  --max-samples 3000

# 4. ブラウザで開く
# 生成された viewer.html をブラウザで開くだけで動作（外部依存なし）
```

#### 特定の打者だけのビューアを生成

`--batter` オプションで打者を指定すると、その打者の全データだけを含むビューアが生成されます。

```bash
# MLBAM ID で指定
python3 -m tools.generate_viewer \
  --pred-dir outputs/.../test/2026-03-20-120000 \
  --batter 660271

# 名前の一部で指定（部分一致検索）
python3 -m tools.generate_viewer \
  --pred-dir outputs/.../test/2026-03-20-120000 \
  --batter Ohtani
```

`--batter` 指定時は自動的に `filter=all`（全サンプル）、`max-samples=100000`（実質無制限）に調整されます。

### ビューア画面

1枚の "カード" に1サンプル（1投球）の全情報がまとまっています:

- **試合情報**: 打者名・投手名（MLBAM ID 付き）、対戦チーム、日付、Game ID、打席番号（メタデータ構築済みの場合）
- **入力情報**: 球種・投手投げ手・打席・カウント・走者/アウト状況・球速・回転数など
- **ストライクゾーン**: SVG で投球コースを描画（ゾーン内/外で色分け）
- **Swing Attempt**: 予測確率バー + GT との正誤
- **Swing Result**: 3クラス確率バー（foul / hit_into_play / miss）
- **BB Type**: 4クラス確率バー（ground_ball / fly_ball / line_drive / popup）
- **Regression**: launch_speed / launch_angle / hit_distance / spray_angle の予測値・GT・誤差

### 操作方法

| 操作 | 説明 |
|---|---|
| `←` / `→` キー | 前後のサンプルに移動 |
| `Home` / `End` | 先頭 / 最後に移動 |
| Prev / Next ボタン | クリックで前後移動 |
| サンプル番号入力 | 直接ジャンプ |
| フィルタドロップダウン | 全て / SA誤分類 / SR誤分類 / BT誤分類 / いずれか誤分類 |

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--pred-dir` | （必須） | `predictions_*.npz` と `predictions_meta_*.json` があるディレクトリ |
| `--split` | `test` | 評価スプリット（`test` / `val`） |
| `--max-samples` | `2000` | HTML に含める最大サンプル数 |
| `--filter` | `random` | サンプル選択フィルタ（`all` / `misclassified_sa` / `misclassified_sr` / `misclassified_bt` / `random` / `include_invalid`） |
| `--sort` | `index` | ソート基準（`index` / `sa_error` / `reg_error`） |
| `--output` | `pred-dir/viewer.html` | 出力 HTML ファイルパス |
| `--seed` | `42` | `random` フィルタ時のシード |
| `--metadata-dir` | `/workspace/datasets/statcast-customized-v2` | メタデータディレクトリ（試合情報・選手名表示用） |
| `--batter` | — | 特定の打者のみ表示（MLBAM ID または名前の一部） |

### データサイズの目安

| サンプル数 | HTML ファイルサイズ |
|---|---|
| 2,000 | 〜3 MB |
| 5,000 | 〜7 MB |
| 10,000 | 〜14 MB |

生成された HTML は CSS・JavaScript・データ全てを内包しており、ブラウザで開くだけで動作します。

---

## plot\_curves — 学習曲線プロット

`train.py` が出力する `history.json` を読み込み、学習曲線をプロットします。

```bash
# 全プロットを生成（デフォルト）
python3 -m tools.plot_curves /workspace/outputs/all/dnn/2026-03-21-022236

# プロット種類を指定
python3 -m tools.plot_curves outputs/run1 --plots total_loss accuracy

# 画像サイズ・フォントサイズを変更
python3 -m tools.plot_curves outputs/run1 --figsize 16 10 --fontsize 14

# PDF で出力
python3 -m tools.plot_curves outputs/run1 --format pdf --dpi 300
```

### 出力されるプロット

| 種類 | ファイル名 | 内容 |
|---|---|---|
| `total_loss` | `loss_total.{fmt}` | Total Loss の train/val 曲線 |
| `individual_loss` | `loss_components.{fmt}` | 各損失コンポーネント（swing_attempt, swing_result, bb_type, regression, physics）の train/val サブプロット |
| `accuracy` | `accuracy.{fmt}` | 分類タスクの検証精度（swing_attempt, swing_result, bb_type） |
| `lr` | `lr.{fmt}` | 学習率スケジュールの推移 |

出力先は `{output_dir}/figs/{タイムスタンプ}/` に自動作成されます。

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `output_dir`（位置引数） | （必須） | `train.py` の出力ディレクトリ（`history.json` が存在するパス） |
| `--plots` | 全種類 | 生成するプロットの種類（`total_loss` / `individual_loss` / `accuracy` / `lr`、複数指定可） |
| `--figsize W H` | `12 8` | 画像サイズ（幅 高さ） |
| `--fontsize` | `12` | フォントサイズ |
| `--format` | `png` | 出力形式（`png` / `pdf` / `svg`） |
| `--dpi` | `150` | 解像度 |
