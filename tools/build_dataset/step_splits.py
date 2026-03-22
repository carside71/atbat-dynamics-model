"""Step 4: 分割・打者履歴・メタデータ構築・保存."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tools.build_dataset.columns import BATTER_HIST_NUM_ATBATS, TRAIN_END, VALID_END


@dataclass
class SplitsReport:
    """分割ステップのレポート."""

    split_sizes: dict[str, int] = field(default_factory=dict)
    split_date_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)
    history_stats: dict[str, float] = field(default_factory=dict)

    def display(self) -> None:

        print("=== Step 4: Splits & Save ===")

        # 分割サイズ
        print("\n  分割サイズ (at_bat_id数):")
        for split, size in self.split_sizes.items():
            dr = self.split_date_ranges.get(split, ("?", "?"))
            print(f"    {split}: {size:,} ({dr[0]} ~ {dr[1]})")

        # 履歴統計
        if self.history_stats:
            print("\n  打者履歴:")
            print(f"    エントリ数: {self.history_stats['entries']:,.0f}")
            print(f"    平均履歴長: {self.history_stats['mean_len']:.1f}")
            print(f"    最大履歴長: {self.history_stats['max_len']:.0f}")


def _temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """game_date による時系列分割."""
    game_date = pd.to_datetime(df["game_date"])
    train_end = pd.Timestamp(TRAIN_END)
    valid_end = pd.Timestamp(VALID_END)

    train_mask = game_date <= train_end
    valid_mask = (game_date > train_end) & (game_date <= valid_end)
    test_mask = game_date > valid_end

    train_ids = df.loc[train_mask, "at_bat_id"].unique()
    valid_ids = df.loc[valid_mask, "at_bat_id"].unique()
    test_ids = df.loc[test_mask, "at_bat_id"].unique()

    return (
        pd.Series(sorted(train_ids), name="at_bat_id"),
        pd.Series(sorted(valid_ids), name="at_bat_id"),
        pd.Series(sorted(test_ids), name="at_bat_id"),
    )


def _build_batter_history(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """打者履歴ルックアップテーブルを構築する.

    Returns:
        (batter_game_history, atbat_row_indices)
    """
    N = BATTER_HIST_NUM_ATBATS

    # at_bat_id → 行インデックスのマッピング
    atbat_groups = df.groupby("at_bat_id").apply(lambda g: g.index.tolist(), include_groups=False)
    atbat_row_indices = pd.DataFrame({"at_bat_id": atbat_groups.index, "row_indices": atbat_groups.values})

    # (batter, game_pk) → at_bat_ids
    atbat_info = df.groupby("at_bat_id").agg(batter=("batter", "first"), game_pk=("game_pk", "first")).reset_index()

    # batter ごとに game_pk 順で打席を蓄積
    batter_games = (
        atbat_info.sort_values(["batter", "game_pk", "at_bat_id"])
        .groupby(["batter", "game_pk"])["at_bat_id"]
        .apply(list)
        .reset_index()
    )

    history_records = []
    for batter, group in tqdm(batter_games.groupby("batter"), desc="Building batter history", leave=False):
        past_ids: list[int] = []
        for _, row in group.iterrows():
            game_pk = row["game_pk"]
            current_ids = row["at_bat_id"]
            # 現在の試合の打席は含めず、過去の打席のみ
            hist = past_ids[-N:] if len(past_ids) > N else past_ids[:]
            history_records.append(
                {
                    "batter": batter,
                    "game_pk": game_pk,
                    "hist_at_bat_ids": hist,
                    "hist_len": len(hist),
                }
            )
            past_ids.extend(current_ids)

    batter_game_history = pd.DataFrame(history_records)
    return batter_game_history, atbat_row_indices


def _build_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """打席メタデータと選手名マッピングを構築する.

    Returns:
        (atbat_metadata, player_names)
    """
    meta_cols = ["at_bat_id", "batter", "game_pk", "game_date"]
    optional_cols = ["pitcher", "home_team", "away_team", "at_bat_number"]
    meta_cols += [c for c in optional_cols if c in df.columns]

    metadata = df[meta_cols].groupby("at_bat_id").first().reset_index()
    metadata = metadata.sort_values("at_bat_id").reset_index(drop=True)

    # 選手名は後から tools/build_metadata.py で MLB API から取得可能
    # ここでは空の辞書を返す
    player_names: dict[str, str] = {}

    return metadata, player_names


def run(
    df: pd.DataFrame,
    stats_tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, SplitsReport]:
    """分割・保存を実行する.

    Args:
        df: step_labels から受け取ったDataFrame
        stats_tables: step_labels で生成された stats テーブル
        output_dir: 出力ディレクトリ

    Returns:
        (保存用DataFrame, レポート)
    """
    from tools.build_dataset.step_labels import save_stats

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = SplitsReport()

    # 時系列分割
    train_ids, valid_ids, test_ids = _temporal_split(df)
    report.split_sizes = {
        "train": len(train_ids),
        "valid": len(valid_ids),
        "test": len(test_ids),
    }

    game_date = pd.to_datetime(df["game_date"])
    for split_name, ids in [("train", train_ids), ("valid", valid_ids), ("test", test_ids)]:
        mask = df["at_bat_id"].isin(set(ids))
        dates = game_date[mask]
        if len(dates) > 0:
            report.split_date_ranges[split_name] = (
                str(dates.min().date()),
                str(dates.max().date()),
            )

    # 分割CSV保存
    train_ids.to_frame().to_csv(output_dir / "train_at_bat_ids.csv", index=False)
    valid_ids.to_frame().to_csv(output_dir / "valid_at_bat_ids.csv", index=False)
    test_ids.to_frame().to_csv(output_dir / "test_at_bat_ids.csv", index=False)

    # 打者履歴構築・保存
    batter_game_history, atbat_row_indices = _build_batter_history(df)
    batter_game_history.to_parquet(output_dir / "batter_game_history.parquet", index=False)
    atbat_row_indices.to_parquet(output_dir / "atbat_row_indices.parquet", index=False)

    report.history_stats = {
        "entries": len(batter_game_history),
        "mean_len": float(batter_game_history["hist_len"].mean()),
        "max_len": float(batter_game_history["hist_len"].max()),
    }

    # メタデータ構築・保存
    metadata, player_names = _build_metadata(df)
    metadata.to_parquet(output_dir / "atbat_metadata.parquet", index=False)
    if player_names:
        with open(output_dir / "player_names.json", "w") as f:
            json.dump(player_names, f, ensure_ascii=False, indent=2)

    # stats CSV 保存
    save_stats(stats_tables, output_dir)

    # メタデータ用カラムを落とし、データ保存
    meta_only_cols = ["pitcher", "home_team", "away_team", "at_bat_number"]
    drop_cols = [c for c in meta_only_cols if c in df.columns]
    df_save = df.drop(columns=drop_cols)

    # game_date を date 型に変換して保存
    if "game_date" in df_save.columns:
        df_save["game_date"] = pd.to_datetime(df_save["game_date"]).dt.date

    df_save.to_parquet(output_dir / "pitches.parquet", index=False)

    return df, report
