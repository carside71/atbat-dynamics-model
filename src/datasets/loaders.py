"""データ読み込み・統計ユーティリティ."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from config import DataConfig


def load_stats(stats_dir: Path) -> dict[str, pd.DataFrame]:
    """stats ディレクトリからラベル対応テーブルを読み込む."""
    stats = {}
    for p in sorted(stats_dir.glob("stats_*.csv")):
        key = p.stem.replace("stats_", "")
        stats[key] = pd.read_csv(p)
    return stats


def get_num_classes(stats: dict[str, pd.DataFrame]) -> dict[str, int]:
    """各カテゴリカル特徴量のクラス数を返す."""
    mapping = {
        "p_throws": "p_throws",
        "pitch_type": "pitch_type",
        "batter": "batter",
        "stand": "stand",
        "swing_result": "swing_result",
        "bb_type": "bb_type",
    }
    num_classes = {}
    for feat, stats_key in mapping.items():
        if stats_key in stats:
            num_classes[feat] = len(stats[stats_key])
    return num_classes


def compute_embedding_dim(num_classes: int, max_dim: int = 50) -> int:
    """カテゴリ数から埋め込み次元を決定するヒューリスティック."""
    return min(max_dim, max(2, int(num_classes**0.25 * 4)))


def load_all_parquet_files(data_dir: Path) -> pd.DataFrame:
    """データを読み込む。pitches.parquet があればそれを、なければ全 parquet を結合."""
    single_file = data_dir / "pitches.parquet"
    if single_file.exists():
        return pd.read_parquet(single_file)
    files = sorted(glob.glob(str(data_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_split_at_bat_ids(split_dir: Path, split: str) -> set[int]:
    """split ディレクトリから指定された分割の at_bat_id 集合を読み込む."""
    filename_map = {
        "train": "train_at_bat_ids.csv",
        "val": "valid_at_bat_ids.csv",
        "valid": "valid_at_bat_ids.csv",
        "test": "test_at_bat_ids.csv",
    }
    if split not in filename_map:
        raise ValueError(f"Unknown split: {split}. Use one of {list(filename_map.keys())}")
    filepath = split_dir / filename_map[split]
    df = pd.read_csv(filepath)
    return set(df["at_bat_id"].tolist())


def compute_normalization_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    """連続値カラムの平均と標準偏差を計算する."""
    stats = {}
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        stats[col] = (float(mean), float(std))
    return stats


def create_dataset(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    norm_stats: dict[str, tuple[float, float]],
    reg_norm_stats: dict[str, tuple[float, float]],
    *,
    max_seq_len: int = 0,
    batter_hist_max_atbats: int = 0,
    batter_hist_max_pitches: int = 10,
    batter_history_dir: Path | None = None,
) -> Dataset:
    """設定に基づき適切なデータセットクラスをインスタンス化する."""
    from datasets.statcast import StatcastDataset
    from datasets.statcast_batter_hist import StatcastBatterHistDataset
    from datasets.statcast_sequence import StatcastSequenceDataset

    if batter_hist_max_atbats > 0:
        return StatcastBatterHistDataset(
            df, data_cfg, max_seq_len, batter_hist_max_atbats, batter_hist_max_pitches,
            norm_stats, reg_norm_stats, batter_history_dir,
        )
    if max_seq_len > 0:
        return StatcastSequenceDataset(df, data_cfg, max_seq_len, norm_stats, reg_norm_stats)
    return StatcastDataset(df, data_cfg, norm_stats, reg_norm_stats)
