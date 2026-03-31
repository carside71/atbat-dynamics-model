# src/datasets/

データセットの読み込み・前処理パッケージ。

## データソース

`/workspace/datasets/statcast-customized-v2/` に配置されたデータセットを使用します。フラット構造で全ファイルが単一ディレクトリに格納されています。

| ファイル | 内容 |
|---|---|
| `pitches.parquet` | 全投球データ（単一ファイル） |
| `stats_*.csv` | カテゴリカル特徴量のラベル対応テーブル |
| `train_at_bat_ids.csv` / `valid_at_bat_ids.csv` / `test_at_bat_ids.csv` | 時系列分割の `at_bat_id` リスト |
| `batter_game_history.parquet` / `atbat_row_indices.parquet` | 打者履歴ルックアップテーブル |
| `pitcher_game_history.parquet` | 投手履歴ルックアップテーブル |
| `atbat_metadata.parquet` / `player_names.json` | ビューア用メタデータ |

データ分割は **時系列分割**（`game_date` ベース）で行われます。ダブルヘッダーを正しく区別するため `game_pk`（試合ごとに一意な ID）を使用します。

`model_scope="outcome"` の場合、分割後に **`swing_attempt=1` のサンプルのみに絞り込み** が行われます。これにより outcome モデルはスイングが発生した投球のみで学習・評価されます（フィルタリングは `train.py` / `test.py` 側で実施、データセットクラス自体は変更なし）。

分割基準日:

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

前処理は `tools/build_dataset/` パッケージにモジュール化されており、`notes/00_build_dataset/build_dataset.ipynb` で実行します。中間ファイルは生成せず、全処理をインメモリで実行して最終結果を直接出力します。

| Step | モジュール | 内容 |
|---|---|---|
| 1. Filter | `step_filter.py` | 2000球以上の打者を抽出 |
| 2. Features | `step_features.py` | カラム選択・ゲームステート・軌道特徴量・plate_z正規化・spray_angle |
| 3. Labels | `step_labels.py` | description解析・swing_result統合(3クラス)・カテゴリカルエンコード・stats生成 |
| 4. Splits | `step_splits.py` | 時系列分割・打者履歴構築・投手履歴構築・メタデータ構築・保存 |
| 5. Quality | `step_validate.py` | ソースデータの品質レポート（構築処理の成否ではない） |

### plate_z ゾーン正規化

plate_z（投球の縦方向ホームプレート通過位置）を打者ごとのストライクゾーンで正規化した `plate_z_norm` を使用しています。

$$
\text{plate\_z\_norm} = \frac{\text{plate\_z} - \text{sz\_bot}}{\text{sz\_top} - \text{sz\_bot}}
$$

- `0` = ストライクゾーン下端、`1` = ストライクゾーン上端
- 0〜1 の範囲外はボールゾーン

### swing_result クラス統合

元の 9 クラスを以下の 3 クラスに統合しています。

| 統合後 | クラスインデックス | 元クラス |
|---|---|---|
| **foul** | 0 | foul, foul_bunt, foul_tip, foul_pitchout |
| **hit_into_play** | 1 | hit_into_play, hit_into_play_score, hit_into_play_no_out, bunt_foul_tip |
| **miss** | 2 | swinging_strike, swinging_strike_blocked, missed_bunt, swinging_pitchout |

### bb_type クラスと物理的定義

打球タイプは Statcast の打出角（launch_angle）基準で以下のように定義されます。`PhysicsLoss` はこの対応関係を利用して分類予測と回帰予測の整合性を強制します。

| クラス | クラスインデックス | launch_angle 範囲（Statcast 基準） |
|---|---|---|
| **ground_ball** | 0 | < 10° |
| **fly_ball** | 1 | 25° < LA ≤ 50° |
| **line_drive** | 2 | 10° ≤ LA ≤ 25° |
| **popup** | 3 | > 50° |

### spray_angle とフェアゾーン

spray_angle（打球方向角度）はフェアゾーンとファウルゾーンの判別に使われます。`PhysicsLoss` では以下の制約を適用します。

| swing_result | spray_angle の期待範囲 |
|---|---|
| **hit_into_play** (cls 1) | -45° ≤ SA ≤ +45°（フェアゾーン） |
| **foul** (cls 0) | SA < -45° or SA > +45°（ファウルゾーン） |

---

## ファイル構成

| ファイル | 内容 |
|---|---|
| `__init__.py` | パッケージの公開 API（再エクスポート） |
| `loaders.py` | データ読み込み・統計ユーティリティ関数 + `create_dataset` ファクトリ |
| `statcast_base.py` | `StatcastBaseDataset`（共通基底クラス） |
| `statcast.py` | `StatcastDataset`（単一投球データセット） |
| `statcast_sequence.py` | `StatcastSequenceDataset`（系列対応データセット） |
| `statcast_batter_hist.py` | `StatcastBatterHistDataset`（打者履歴・投手履歴対応データセット） |

### クラス継承構造

```
StatcastBaseDataset (statcast_base.py)
├── StatcastDataset (statcast.py)
├── StatcastSequenceDataset (statcast_sequence.py)
└── StatcastBatterHistDataset (statcast_batter_hist.py)  ← 投手履歴にも対応
```

`StatcastBaseDataset` は全データセット共通の以下の処理を提供します:

- **特徴量初期化**: カテゴリカル・連続値（z-score 正規化）・順序特徴量のテンソル化
- **ターゲット初期化**: swing_attempt, swing_result, bb_type, 回帰ターゲット（正規化 + マスク）
- **`_base_item(idx)`**: 共通のサンプル辞書構築
- **`_build_atbat_groups(at_bat_ids)`**: at_bat_id によるグルーピング（Sequence / BatterHist で使用）
- **`_build_seq_features(past_indices, max_seq_len)`**: 打席内投球シーケンスの特徴量構築（Sequence / BatterHist で使用）

各サブクラスは基底クラスを継承し、固有の機能のみを追加します。

---

## loaders.py — ユーティリティ関数

| 関数 | 説明 |
|---|---|
| `load_stats(stats_dir)` | `stats_*.csv` ファイルからラベル対応テーブルを読み込み |
| `get_num_classes(stats)` | 各カテゴリカル特徴量のクラス数を取得 |
| `compute_embedding_dim(num_classes)` | カテゴリ数から埋め込み次元を決定するヒューリスティック |
| `load_all_parquet_files(data_dir)` | `pitches.parquet`（単一ファイル）を読み込み。存在しない場合は全 parquet を結合 |
| `load_split_at_bat_ids(split_dir, split)` | train/valid/test の `at_bat_id` 集合を読み込み |
| `compute_normalization_stats(df, columns)` | 指定カラムの平均・標準偏差を計算 |
| `create_dataset(df, data_cfg, ...)` | 設定に基づき適切なデータセットクラスを自動選択・インスタンス化 |

---

## StatcastDataset

`StatcastBaseDataset` を継承。**各投球を独立したサンプルとして扱う**標準データセット。

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
│   pfx_x, pfx_z, plate_x, plate_z,               │
│   vx0, vy0, vz0, ax, ay, az,                    │
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
│                hit_distance_sc, spray_angle)    │
│   reg_mask (各回帰ターゲットの有効フラグ)            │
└─────────────────────────────────────────────────┘
```

### 使い方

```python
from datasets import StatcastDataset, load_all_parquet_files, compute_normalization_stats

df = load_all_parquet_files(data_dir)
norm_stats = compute_normalization_stats(df, continuous_features)
ds = StatcastDataset(df, data_cfg, norm_stats, reg_norm_stats)

# または create_dataset ファクトリを使用（train.py / test.py と同じ方法）
from datasets import create_dataset
ds = create_dataset(df, data_cfg, norm_stats, reg_norm_stats)  # max_seq_len=0 でデフォルト
```

---

## StatcastSequenceDataset

`StatcastBaseDataset` を継承。**同一打席（at_bat_id）内の過去投球を系列として提供する**データセット。
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

# または create_dataset ファクトリを使用
from datasets import create_dataset
ds = create_dataset(df, data_cfg, norm_stats, reg_norm_stats, max_seq_len=10)
```

---

## StatcastBatterHistDataset

`StatcastBaseDataset` を継承。**打者・投手の過去打席履歴（生投球データ）を提供する**データセット。
`StatcastSequenceDataset` の全機能に加え、当該試合以前の過去 N 打席分の Statcast 生データを返す。投手履歴も同時に有効化可能。

### 仕組み

```
打者 A の試合履歴 (game_pk 順)
────────────────────────────────
  game_pk=100 (4 打席) → at_bat_id: [a1, a2, a3, a4]
  game_pk=200 (3 打席) → at_bat_id: [a5, a6, a7]
  game_pk=300 (5 打席) → 現在の試合

現在の試合の打席では、過去 N=50 打席（a1〜a7 + それ以前）の
投球データを参照可能。ダブルヘッダーも game_pk で区別。

投手履歴も同じ仕組み（pitcher, game_pk でグルーピング）:

投手 X の試合履歴 (game_pk 順)
────────────────────────────────
  game_pk=100 (8 打席) → at_bat_id: [p1, p2, ..., p8]
  game_pk=200 (7 打席) → at_bat_id: [p9, ..., p15]
  game_pk=300 (6 打席) → 現在の試合

現在の試合の打席では、投手の過去 N=50 打席分の
被打球データを参照可能。
```

### データソース

データセットディレクトリ内の以下のファイルを使用（`tools/build_dataset/step_splits.py` で生成）:

| ファイル | 内容 |
|---|---|
| `batter_game_history.parquet` | (batter, game_pk) → 打者の過去 N 打席の at_bat_id リスト |
| `pitcher_game_history.parquet` | (pitcher, game_pk) → 投手の過去 N 打席の at_bat_id リスト |
| `atbat_row_indices.parquet` | at_bat_id → 元データの行インデックス |

### 追加の出力フィールド

`StatcastSequenceDataset` の出力に加えて、以下の打者履歴フィールドを返す（`batter_hist_max_atbats > 0` 時）:

| フィールド | 形状 | 説明 |
|---|---|---|
| `hist_pitch_type` | `(N, P)` | 過去打席の投球球種 |
| `hist_cont` | `(N, P, 15)` | 過去打席の投球連続値特徴量（正規化済み） |
| `hist_swing_attempt` | `(N, P)` | 過去打席のスイング有無 |
| `hist_swing_result` | `(N, P)` | 過去打席のスイング結果 |
| `hist_bb_type` | `(N,)` | 打球種別（ground_ball 等） |
| `hist_launch_speed` | `(N,)` | 打球速度（正規化済み） |
| `hist_launch_angle` | `(N,)` | 打球角度（正規化済み） |
| `hist_spray_angle` | `(N,)` | 打球方向角度（正規化済み） |
| `hist_pitch_mask` | `(N, P)` | 投球レベルの有効マスク |
| `hist_atbat_mask` | `(N,)` | 打席レベルの有効マスク |

> **N** = `batter_hist_max_atbats`（デフォルト 50）、**P** = `batter_hist_max_pitches`（デフォルト 10）

さらに、以下の投手履歴フィールドを返す（`pitcher_hist_max_atbats > 0` 時）:

| フィールド | 形状 | 説明 |
|---|---|---|
| `pitcher_hist_pitch_type` | `(N', P')` | 投手の過去対戦の投球球種 |
| `pitcher_hist_cont` | `(N', P', 15)` | 投手の過去対戦の投球連続値特徴量（正規化済み） |
| `pitcher_hist_swing_attempt` | `(N', P')` | 投手の過去対戦のスイング有無 |
| `pitcher_hist_swing_result` | `(N', P')` | 投手の過去対戦のスイング結果 |
| `pitcher_hist_bb_type` | `(N',)` | 打球種別（ground_ball 等） |
| `pitcher_hist_launch_speed` | `(N',)` | 被打球速度（正規化済み） |
| `pitcher_hist_launch_angle` | `(N',)` | 被打球角度（正規化済み） |
| `pitcher_hist_spray_angle` | `(N',)` | 被打球方向角度（正規化済み） |
| `pitcher_hist_pitch_mask` | `(N', P')` | 投球レベルの有効マスク |
| `pitcher_hist_atbat_mask` | `(N',)` | 打席レベルの有効マスク |

> **N'** = `pitcher_hist_max_atbats`（デフォルト 50）、**P'** = `pitcher_hist_max_pitches`（デフォルト 10）

> **注意**: 時系列分割が必須です。ランダム分割では将来のデータが履歴に混入するリークが発生します。
> **注意**: 投手履歴を有効化するには `pitches.parquet` に `pitcher` カラムが含まれている必要があります（`build_dataset` の再実行が必要な場合があります）。

### 使い方

```python
from datasets import StatcastBatterHistDataset

# 打者履歴のみ
ds = StatcastBatterHistDataset(
    df, data_cfg, max_seq_len=10, norm_stats=norm_stats,
    batter_hist_max_atbats=50,
    batter_hist_max_pitches=10,
)

# 打者履歴 + 投手履歴
ds = StatcastBatterHistDataset(
    df, data_cfg, max_seq_len=10,
    batter_hist_max_atbats=50,
    batter_hist_max_pitches=10,
    pitcher_hist_max_atbats=50,
    pitcher_hist_max_pitches=10,
    norm_stats=norm_stats,
)

# または create_dataset ファクトリを使用
from datasets import create_dataset
ds = create_dataset(
    df, data_cfg, norm_stats, reg_norm_stats,
    max_seq_len=10,
    batter_hist_max_atbats=50, batter_hist_max_pitches=10,
    pitcher_hist_max_atbats=50, pitcher_hist_max_pitches=10,
)
```
