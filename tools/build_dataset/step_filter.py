"""Step 1: 打者フィルタ — 2000球以上の打者のみ抽出."""

from __future__ import annotations

import glob
from dataclasses import dataclass, field

import pandas as pd

from tools.build_dataset.columns import MIN_PITCHES, RAW_COLUMNS


@dataclass
class FilterReport:
    """フィルタステップのレポート."""

    total_batters: int = 0
    qualified_batters: int = 0
    total_pitches_before: int = 0
    total_pitches_after: int = 0
    pitch_counts: pd.Series = field(default_factory=pd.Series)

    def display(self) -> None:
        """ノートブックでの表示."""
        import matplotlib.pyplot as plt
        from IPython.display import display as ipy_display

        print(f"=== Step 1: Filter (min {MIN_PITCHES} pitches) ===")
        print(f"  打者数: {self.total_batters:,} → {self.qualified_batters:,}")
        print(f"  投球数: {self.total_pitches_before:,} → {self.total_pitches_after:,}")
        print(f"  保持率: {self.total_pitches_after / self.total_pitches_before:.1%}")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(self.pitch_counts.values, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(MIN_PITCHES, color="red", linestyle="--", label=f"threshold={MIN_PITCHES}")
        ax.set_xlabel("num pitches")
        ax.set_ylabel("num batters")
        ax.set_title("distribution of pitches per batter (qualified)")
        ax.legend()
        fig.tight_layout()
        ipy_display(fig)
        plt.close(fig)


def run(source_dir: str) -> tuple[pd.DataFrame, FilterReport]:
    """生CSVからデータを読み込み、打者フィルタを適用する.

    Args:
        source_dir: 生Statcast CSVファイルのディレクトリ

    Returns:
        (フィルタ済みDataFrame, レポート)
    """
    csv_files = sorted(glob.glob(f"{source_dir}/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    # 全CSV読み込み（必要カラムのみ）
    available_cols = set(pd.read_csv(csv_files[0], nrows=0).columns)
    use_cols = [c for c in RAW_COLUMNS if c in available_cols]

    dfs = [pd.read_csv(f, usecols=use_cols, low_memory=False) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    report = FilterReport()
    report.total_pitches_before = len(df)

    # 打者ごとの投球数カウント
    batter_counts = df["batter"].value_counts()
    report.total_batters = len(batter_counts)

    # フィルタ
    qualified = batter_counts[batter_counts >= MIN_PITCHES]
    report.qualified_batters = len(qualified)
    report.pitch_counts = qualified

    batter_set = set(qualified.index)
    df = df[df["batter"].isin(batter_set)].reset_index(drop=True)
    report.total_pitches_after = len(df)

    return df, report
