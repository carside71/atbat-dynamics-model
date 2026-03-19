"""Phase 0 & 1: game_pk/game_date を最終 parquet に追加し、時系列分割と打者履歴テーブルを構築する.

Usage:
    python scripts/add_game_info_and_rebuild.py

処理内容:
  1. preprocess_01 から game_pk, game_date を読み込み、最終 data parquet に追加
  2. game_date ベースで train/valid/test を時系列分割
  3. 打者履歴テーブル（batter × game_pk → 直近50打席の at_bat_id リスト）を構築
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

# === 設定 ===
PREPROCESS_01_DIR = Path("/workspace/datasets/statcast-customized-tmp/preprocess_01")
DATA_DIR = Path("/workspace/datasets/statcast-customized/data")
SPLIT_DIR = Path("/workspace/datasets/statcast-customized/split")
HISTORY_DIR = Path("/workspace/datasets/statcast-customized/batter_history")

# 時系列分割の日付境界
TRAIN_END = "2024-06-30"  # Train: ~ 2024-06-30
VALID_END = "2024-10-31"  # Valid: 2024-07-01 ~ 2024-10-31
# Test: 2025-01-01 ~

BATTER_HIST_NUM_ATBATS = 50  # 直近何打席分の履歴を保持するか


def step1_add_game_info() -> pd.DataFrame:
    """preprocess_01 から game_pk, game_date を最終 parquet に追加."""
    print("=" * 60)
    print("Step 1: Adding game_pk and game_date to final parquet files")
    print("=" * 60)

    p01_files = sorted(PREPROCESS_01_DIR.glob("*.parquet"))
    all_dfs = []

    for p01_path in tqdm(p01_files, desc="Processing files"):
        data_path = DATA_DIR / p01_path.name
        if not data_path.exists():
            print(f"  WARNING: {data_path.name} not found in data dir, skipping")
            continue

        df_p01 = pd.read_parquet(p01_path, columns=["game_pk", "game_date"])
        df_data = pd.read_parquet(data_path)

        assert len(df_p01) == len(df_data), (
            f"Row count mismatch for {p01_path.name}: preprocess_01={len(df_p01)}, data={len(df_data)}"
        )

        # game_date を datetime に変換して日付のみ保持
        df_data["game_pk"] = df_p01["game_pk"].values
        df_data["game_date"] = pd.to_datetime(df_p01["game_date"].values).date

        # 上書き保存
        df_data.to_parquet(data_path, index=False)
        all_dfs.append(df_data)

    all_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total samples: {len(all_df):,}")
    print(f"  Date range: {all_df['game_date'].min()} ~ {all_df['game_date'].max()}")
    print(f"  Unique game_pk: {all_df['game_pk'].nunique():,}")
    print(f"  Unique batters: {all_df['batter'].nunique():,}")
    return all_df


def step2_temporal_split(all_df: pd.DataFrame) -> None:
    """game_date ベースで時系列分割を行い、at_bat_id リストを保存."""
    print()
    print("=" * 60)
    print("Step 2: Temporal train/valid/test split")
    print("=" * 60)

    train_end = pd.Timestamp(TRAIN_END).date()
    valid_end = pd.Timestamp(VALID_END).date()

    game_date = all_df["game_date"]
    train_mask = game_date <= train_end
    valid_mask = (game_date > train_end) & (game_date <= valid_end)
    test_mask = game_date > valid_end

    train_ids = all_df.loc[train_mask, "at_bat_id"].unique()
    valid_ids = all_df.loc[valid_mask, "at_bat_id"].unique()
    test_ids = all_df.loc[test_mask, "at_bat_id"].unique()

    print(f"  Train: {len(train_ids):,} at-bats ({train_mask.sum():,} pitches) [~ {TRAIN_END}]")
    print(f"  Valid: {len(valid_ids):,} at-bats ({valid_mask.sum():,} pitches) [{TRAIN_END} ~ {VALID_END}]")
    print(f"  Test:  {len(test_ids):,} at-bats ({test_mask.sum():,} pitches) [{VALID_END} ~]")

    # リーク検証: 日付範囲の重複がないことを確認
    train_dates = all_df.loc[train_mask, "game_date"]
    valid_dates = all_df.loc[valid_mask, "game_date"]
    test_dates = all_df.loc[test_mask, "game_date"]
    assert train_dates.max() <= valid_dates.min(), "Train/Valid date overlap!"
    assert valid_dates.max() <= test_dates.min(), "Valid/Test date overlap!"
    print("  ✓ No date overlap between splits")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(train_ids, name="at_bat_id").to_csv(SPLIT_DIR / "train_at_bat_ids.csv", index=False)
    pd.Series(valid_ids, name="at_bat_id").to_csv(SPLIT_DIR / "valid_at_bat_ids.csv", index=False)
    pd.Series(test_ids, name="at_bat_id").to_csv(SPLIT_DIR / "test_at_bat_ids.csv", index=False)
    print(f"  Saved split files to: {SPLIT_DIR}")


def step3_build_batter_history(all_df: pd.DataFrame) -> None:
    """打者履歴テーブルを構築する.

    各 (batter, game_pk) の組み合わせに対して、
    その試合より前の直近 N 打席の at_bat_id リストを作成する。

    ダブルヘッダー対策:
      同じ日に同じ打者が複数の game_pk に出場する場合がある。
      game_pk で個別に扱い、同じ game_pk 内の打席は履歴に含めない。
      ダブルヘッダーの2試合目では1試合目の結果も履歴に含める。
      game_pk の大小関係で試合順序を決定する（MLB の game_pk は時系列順）。
    """
    print()
    print("=" * 60)
    print("Step 3: Building batter history lookup table")
    print("=" * 60)

    # 打席テーブル: 各 at_bat_id の (batter, game_pk, game_date) + 行インデックス範囲
    # at_bat_id ごとに最初の行の情報を取得
    atbat_info = (
        all_df.groupby("at_bat_id")
        .agg(
            batter=("batter", "first"),
            game_pk=("game_pk", "first"),
            game_date=("game_date", "first"),
            row_start=("at_bat_id", lambda x: x.index.min()),
            row_end=("at_bat_id", lambda x: x.index.max()),
        )
        .reset_index()
        .sort_values(["batter", "game_pk", "at_bat_id"])
        .reset_index(drop=True)
    )
    print(f"  Total at-bats: {len(atbat_info):,}")
    print(f"  Unique batters: {atbat_info['batter'].nunique():,}")

    # 各 (batter, game_pk) に対して、その試合より前の直近50打席を計算
    # batter ごとに処理
    history_records = []
    batters = atbat_info["batter"].unique()

    for batter_id in tqdm(batters, desc="Building history per batter"):
        batter_atbats = atbat_info[atbat_info["batter"] == batter_id].copy()
        # game_pk で試合順にソート（MLB の game_pk は時系列順）
        batter_atbats = batter_atbats.sort_values(["game_pk", "at_bat_id"]).reset_index(drop=True)

        # この打者が出場した各 game_pk について、それより前の打席を取得
        game_pks = batter_atbats["game_pk"].unique()

        for gpk in game_pks:
            # この試合より前の打席（game_pk が小さい = 時系列的に前）
            past_atbats = batter_atbats[batter_atbats["game_pk"] < gpk]
            past_ids = past_atbats["at_bat_id"].values

            # 直近 N 打席のみ保持
            if len(past_ids) > BATTER_HIST_NUM_ATBATS:
                past_ids = past_ids[-BATTER_HIST_NUM_ATBATS:]

            history_records.append(
                {
                    "batter": batter_id,
                    "game_pk": gpk,
                    "hist_at_bat_ids": past_ids.tolist(),
                    "hist_len": len(past_ids),
                }
            )

    history_df = pd.DataFrame(history_records)
    print(f"  Total (batter, game_pk) combinations: {len(history_df):,}")
    print("  History length stats:")
    print(f"    Mean: {history_df['hist_len'].mean():.1f}")
    print(f"    Median: {history_df['hist_len'].median():.0f}")
    print(f"    Min: {history_df['hist_len'].min()}")
    print(f"    Max: {history_df['hist_len'].max()}")
    print(f"    Zero history: {(history_df['hist_len'] == 0).sum():,}")

    # 打席→行インデックスのマッピングも保存
    atbat_to_rows = all_df.groupby("at_bat_id").apply(lambda g: g.index.tolist(), include_groups=False).reset_index()
    atbat_to_rows.columns = ["at_bat_id", "row_indices"]

    # 保存
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_parquet(HISTORY_DIR / "batter_game_history.parquet", index=False)
    atbat_to_rows.to_parquet(HISTORY_DIR / "atbat_row_indices.parquet", index=False)
    print(f"  Saved history files to: {HISTORY_DIR}")


if __name__ == "__main__":
    all_df = step1_add_game_info()
    step2_temporal_split(all_df)
    step3_build_batter_history(all_df)
    print()
    print("Done!")
