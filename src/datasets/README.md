# src/datasets/

データセットの読み込み・前処理パッケージ。

## データソース

`/workspace/datasets/statcast-customized/` に配置された Statcast データを使用します。

| ディレクトリ | 内容 |
|---|---|
| `data/` | 年月別の Parquet ファイル（例: `statcast_2024_04.parquet`） |
| `stats/` | カテゴリカル特徴量のラベル対応テーブル（CSV） |
| `split/` | 学習/検証/テスト分割の `at_bat_id` リスト（CSV）※時系列分割 |
| `batter_history/` | 打者履歴ルックアップテーブル（Parquet） |

データ分割は **時系列分割**（`game_date` ベース）で行われます。ダブルヘッダーを正しく区別するため `game_pk`（試合ごとに一意な ID）を使用します。分割基準日:

| 分割 | 期間 |
|---|---|
| Train | 〜 2024-06-30 |
| Valid | 2024-07-01 〜 2024-10-30 |
| Test | 2025-03-15 〜 |

---

## 入力特徴量

モデルへの入力は以下の3種に分類されます。

| 種別 | 特徴量 |
|---|---|
| カテゴリカル | p_throws, pitch_type, batter, stand, base_out_state, count_state |
| 連続値 | release_speed, release_spin_rate, pfx_x, pfx_z, plate_x, plate_z, vx0, vy0, vz0, ax, ay, az, sz_top, sz_bot, plate_z_norm |
| 順序値 | inning_clipped, is_inning_top, diff_score_clipped, pitch_number_clipped |

連続値特徴量には投球軌道パラメータ（vx0, vy0, vz0, ax, ay, az）、ストライクゾーン上下端（sz_top, sz_bot）、およびゾーン正規化済み縦位置（plate_z_norm）を含みます。

---

## 前処理パイプライン

前処理は `notes/00_build_dataset/` 配下のノートブックで段階的に実行します。中間データは `/workspace/datasets/statcast-customized-tmp/` に保存し、最終結果を `/workspace/datasets/statcast-customized/` にコピーして運用します。

| ステップ | ノートブック | 内容 |
|---|---|---|
| 01 | `00_prepare_dataset_01.ipynb` | 打席数 ≥ 2000 の打者を抽出 |
| 02 | `01_prepare_dataset_02.ipynb` | 必要カラムの選択 |
| 03 | `02_prepare_dataset_03.ipynb` | 特徴量エンジニアリング（カウント状態・走者アウト状態の統合等） |
| 04 | `03_prepare_dataset_04.ipynb` | スイング関連カラムの整備 |
| 05 | `04_prepare_dataset_05.ipynb` | 投球軌道特徴量（vx0〜az, sz_top, sz_bot）の追加 |
| 06 | `06_consolidate_swing_result.ipynb` | swing_result を 9 クラスから 3 クラスに統合 |
| 07 | `07_normalize_plate_z.ipynb` | plate_z をストライクゾーンで正規化し plate_z_norm を追加 |

### plate_z ゾーン正規化

plate_z（投球の縦方向ホームプレート通過位置）を打者ごとのストライクゾーンで正規化した `plate_z_norm` を使用しています。

$$
\text{plate\_z\_norm} = \frac{\text{plate\_z} - \text{sz\_bot}}{\text{sz\_top} - \text{sz\_bot}}
$$

- `0` = ストライクゾーン下端、`1` = ストライクゾーン上端
- 0〜1 の範囲外はボールゾーン

### swing_result クラス統合

元の 9 クラスを以下の 3 クラスに統合しています。

| 統合後 | 元クラス |
|---|---|
| **foul** | foul, foul_bunt, foul_tip, foul_pitchout |
| **hit_into_play** | hit_into_play, hit_into_play_score, hit_into_play_no_out, bunt_foul_tip |
| **miss** | swinging_strike, swinging_strike_blocked, missed_bunt, swinging_pitchout |

---

## ファイル構成

| ファイル | 内容 |
|---|---|
| `__init__.py` | パッケージの公開 API（再エクスポート） |
| `loaders.py` | データ読み込み・統計ユーティリティ関数 |
| `statcast.py` | `StatcastDataset`（単一投球データセット） |
| `statcast_sequence.py` | `StatcastSequenceDataset`（系列対応データセット） |
| `statcast_batter_hist.py` | `StatcastBatterHistDataset`（打者履歴対応データセット） |

---

## loaders.py — ユーティリティ関数

| 関数 | 説明 |
|---|---|
| `load_stats(stats_dir)` | `stats/` ディレクトリからラベル対応テーブル（CSV）を読み込み |
| `get_num_classes(stats)` | 各カテゴリカル特徴量のクラス数を取得 |
| `compute_embedding_dim(num_classes)` | カテゴリ数から埋め込み次元を決定するヒューリスティック |
| `load_all_parquet_files(data_dir)` | `data/` 配下の全 parquet を結合読み込み |
| `load_split_at_bat_ids(split_dir, split)` | train/valid/test の `at_bat_id` 集合を読み込み |
| `compute_normalization_stats(df, columns)` | 指定カラムの平均・標準偏差を計算 |

---

## StatcastDataset

**各投球を独立したサンプルとして扱う**標準データセット。

### 入力特徴量

```
┌─────────────────────────────────────────────────┐
│              StatcastDataset[i]                 │
├─────────────────────────────────────────────────┤
│ カテゴリカル特徴量 (int64)                         │
│   p_throws, pitch_type, batter, stand,          │
│   base_out_state, count_state                   │
├─────────────────────────────────────────────────┤
│ 連続値特徴量 (float32, z-score 正規化)             │
│   release_speed, release_spin_rate,             │
│   pfx_x, pfx_z, plate_x, plate_z,              │
│   vx0, vy0, vz0, ax, ay, az,                   │
│   sz_top, sz_bot, plate_z_norm                  │
├─────────────────────────────────────────────────┤
│ 順序特徴量 (float32)                              │
│   inning_clipped, is_inning_top,                │
│   diff_score_clipped, pitch_number_clipped      │
├─────────────────────────────────────────────────┤
│ ターゲット                                        │
│   swing_attempt (0/1), swing_result (0-8 / -1), │
│   bb_type (0-3 / -1),                           │
│   reg_targets (launch_speed, launch_angle,      │
│                hit_distance_sc)                 │
│   reg_mask (各回帰ターゲットの有効フラグ)            │
└─────────────────────────────────────────────────┘
```

### 使い方

```python
from datasets import StatcastDataset, load_all_parquet_files, compute_normalization_stats

df = load_all_parquet_files(data_dir)
norm_stats = compute_normalization_stats(df, continuous_features)
ds = StatcastDataset(df, data_cfg, norm_stats, reg_norm_stats)
```

---

## StatcastSequenceDataset

**同一打席（at_bat_id）内の過去投球を系列として提供する**データセット。
各サンプルは「現在の投球の特徴量」に加えて「過去投球の系列データ」を含む。

### 仕組み

```
打席 at_bat_id=42 (5球)
──────────────────────────────────────────────────
  pitch 1 → seq_mask: [0,0,...,0]   (過去なし)
  pitch 2 → seq_mask: [1,0,...,0]   (過去1球)
  pitch 3 → seq_mask: [1,1,0,...,0] (過去2球)
  pitch 4 → seq_mask: [1,1,1,0..,0] (過去3球)
  pitch 5 → seq_mask: [1,1,1,1,0..,0] (過去4球)
```

- 系列長は `max_seq_len`（デフォルト 10）で上限を設定
- 系列長未満は右側ゼロパディング、`seq_mask` で有効位置を識別
- `at_bat_id` 列でグルーピングし、行の出現順序＝投球順序として利用

### 追加の出力フィールド

| フィールド | 形状 | 説明 |
|---|---|---|
| `seq_pitch_type` | `(T,)` | 過去投球の球種 |
| `seq_cont` | `(T, num_cont)` | 過去投球の連続特徴量（正規化済み） |
| `seq_swing_attempt` | `(T,)` | 過去投球のスイング有無 |
| `seq_swing_result` | `(T,)` | 過去投球のスイング結果（-1 = スイングなし） |
| `seq_mask` | `(T,)` | 有効マスク（1 = 有効, 0 = パディング） |

> **注意**: 現在の投球のターゲット（`swing_attempt` 等）は系列入力に含まれません（情報リーク防止）。

### 使い方

```python
from datasets import StatcastSequenceDataset

# at_bat_id 列を保持したままの DataFrame を渡す
ds = StatcastSequenceDataset(df, data_cfg, max_seq_len=10, norm_stats=norm_stats)
```

---

## StatcastBatterHistDataset

**打者の過去打席履歴（生投球データ）を提供する**データセット。
`StatcastSequenceDataset` の全機能に加え、当該試合以前の過去 N 打席分の Statcast 生データを返す。

### 仕組み

```
打者 A の試合履歴 (game_pk 順)
────────────────────────────────
  game_pk=100 (4 打席) → at_bat_id: [a1, a2, a3, a4]
  game_pk=200 (3 打席) → at_bat_id: [a5, a6, a7]
  game_pk=300 (5 打席) → 現在の試合

現在の試合の打席では、過去 N=50 打席（a1〜a7 + それ以前）の
投球データを参照可能。ダブルヘッダーも game_pk で区別。
```

### データソース

`batter_history/` ディレクトリの以下のファイルを使用（`scripts/add_game_info_and_rebuild.py` で生成）:

| ファイル | 内容 |
|---|---|
| `batter_game_history.parquet` | (batter, game_pk) → 過去 N 打席の at_bat_id リスト |
| `atbat_row_indices.parquet` | at_bat_id → 元データの行インデックス |

### 追加の出力フィールド

`StatcastSequenceDataset` の出力に加えて、以下の打者履歴フィールドを返す:

| フィールド | 形状 | 説明 |
|---|---|---|
| `hist_pitch_type` | `(N, P)` | 過去打席の投球球種 |
| `hist_cont` | `(N, P, 15)` | 過去打席の投球連続値特徴量（正規化済み） |
| `hist_swing_attempt` | `(N, P)` | 過去打席のスイング有無 |
| `hist_swing_result` | `(N,)` | 打席最終球のスイング結果 |
| `hist_bb_type` | `(N,)` | 打球種別（ground_ball 等） |
| `hist_launch_speed` | `(N,)` | 打球速度（正規化済み） |
| `hist_launch_angle` | `(N,)` | 打球角度（正規化済み） |
| `hist_pitch_mask` | `(N, P)` | 投球レベルの有効マスク |
| `hist_atbat_mask` | `(N,)` | 打席レベルの有効マスク |

> **N** = `batter_hist_max_atbats`（デフォルト 50）、**P** = `batter_hist_max_pitches`（デフォルト 10）

> **注意**: 時系列分割が必須です。ランダム分割では将来のデータが履歴に混入するリークが発生します。

### 使い方

```python
from datasets import StatcastBatterHistDataset

ds = StatcastBatterHistDataset(
    df, data_cfg, max_seq_len=10, norm_stats=norm_stats,
    batter_history_dir="/path/to/batter_history",
    batter_hist_max_atbats=50,
    batter_hist_max_pitches=10,
)
```
