"""データセットの読み込みと前処理."""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def load_parquet_files(data_dir: Path, years: list[int]) -> pd.DataFrame:
    """指定年度の parquet ファイルを結合して読み込む."""
    files = []
    for year in years:
        pattern = str(data_dir / f"statcast_{year}_*.parquet")
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No parquet files found for years {years} in {data_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


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


class StatcastDataset(Dataset):
    """Statcast データセット."""

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DataConfig,
        norm_stats: dict[str, tuple[float, float]] | None = None,
        reg_norm_stats: dict[str, tuple[float, float]] | None = None,
    ):
        self.cfg = cfg
        self.norm_stats = norm_stats or {}
        self.reg_norm_stats = reg_norm_stats or {}

        # === カテゴリカル特徴量 ===
        self.cat_features = {}
        for col in cfg.categorical_features:
            vals = df[col].to_numpy(dtype=np.int64, na_value=-1)
            self.cat_features[col] = torch.from_numpy(vals)

        # === 連続値特徴量（正規化） ===
        cont_arrays = []
        for col in cfg.continuous_features:
            vals = df[col].to_numpy(dtype=np.float32)
            if col in self.norm_stats:
                mean, std = self.norm_stats[col]
                vals = (vals - mean) / std
            vals = np.nan_to_num(vals, nan=0.0)
            cont_arrays.append(vals)
        self.cont_features = torch.from_numpy(
            np.column_stack(cont_arrays) if cont_arrays else np.empty((len(df), 0), dtype=np.float32)
        )

        # === 順序特徴量（float にキャスト） ===
        ord_arrays = []
        for col in cfg.ordinal_features:
            vals = df[col].to_numpy(dtype=np.float32)
            ord_arrays.append(vals)
        self.ord_features = torch.from_numpy(
            np.column_stack(ord_arrays) if ord_arrays else np.empty((len(df), 0), dtype=np.float32)
        )

        # === ターゲット ===
        # swing_attempt: bool → 0/1
        self.swing_attempt = torch.from_numpy(df[cfg.target_cls_swing_attempt].to_numpy(dtype=np.float32))

        # swing_result: Int64, NA → -1
        sr = df[cfg.target_cls_swing_result].to_numpy(dtype=np.int64, na_value=-1)
        self.swing_result = torch.from_numpy(sr)

        # bb_type: Int64, NA → -1
        bt = df[cfg.target_cls_bb_type].to_numpy(dtype=np.int64, na_value=-1)
        self.bb_type = torch.from_numpy(bt)

        # 回帰ターゲット（正規化）
        reg_arrays = []
        reg_mask_arrays = []
        for col in cfg.target_reg:
            vals = df[col].to_numpy(dtype=np.float32)
            mask = ~np.isnan(vals)
            if col in self.reg_norm_stats:
                mean, std = self.reg_norm_stats[col]
                vals = np.where(mask, (vals - mean) / std, 0.0)
            else:
                vals = np.nan_to_num(vals, nan=0.0)
            reg_arrays.append(vals)
            reg_mask_arrays.append(mask.astype(np.float32))

        self.reg_targets = torch.from_numpy(np.column_stack(reg_arrays))
        self.reg_mask = torch.from_numpy(np.column_stack(reg_mask_arrays))

    def __len__(self) -> int:
        return len(self.swing_attempt)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cat = {col: self.cat_features[col][idx] for col in self.cfg.categorical_features}
        return {
            **cat,
            "cont": self.cont_features[idx],
            "ord": self.ord_features[idx],
            "swing_attempt": self.swing_attempt[idx],
            "swing_result": self.swing_result[idx],
            "bb_type": self.bb_type[idx],
            "reg_targets": self.reg_targets[idx],
            "reg_mask": self.reg_mask[idx],
        }
