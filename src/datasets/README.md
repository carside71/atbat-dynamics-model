# src/datasets/

データセットの読み込み・前処理パッケージ。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `__init__.py` | パッケージの公開 API（再エクスポート） |
| `loaders.py` | データ読み込み・統計ユーティリティ関数 |
| `statcast.py` | `StatcastDataset`（単一投球データセット） |
| `statcast_sequence.py` | `StatcastSequenceDataset`（系列対応データセット） |

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
│              StatcastDataset[i]                  │
├─────────────────────────────────────────────────┤
│ カテゴリカル特徴量 (int64)                        │
│   p_throws, pitch_type, batter, stand,          │
│   base_out_state, count_state                   │
├─────────────────────────────────────────────────┤
│ 連続値特徴量 (float32, z-score 正規化)            │
│   release_speed, release_spin_rate,             │
│   pfx_x, pfx_z, plate_x, plate_z               │
├─────────────────────────────────────────────────┤
│ 順序特徴量 (float32)                             │
│   inning_clipped, is_inning_top,                │
│   diff_score_clipped, pitch_number_clipped      │
├─────────────────────────────────────────────────┤
│ ターゲット                                       │
│   swing_attempt (0/1), swing_result (0-8 / -1), │
│   bb_type (0-3 / -1),                           │
│   reg_targets (launch_speed, launch_angle,      │
│                hit_distance_sc)                  │
│   reg_mask (各回帰ターゲットの有効フラグ)           │
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
