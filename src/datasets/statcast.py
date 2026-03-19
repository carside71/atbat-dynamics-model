"""Statcast データセット."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DataConfig


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
