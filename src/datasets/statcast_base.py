"""Statcast データセット基底クラス."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DataConfig


class StatcastBaseDataset(Dataset):
    """全 Statcast データセットの共通初期化・アクセスロジック."""

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
        self.num_cont = self.cont_features.shape[1]

        # === 順序特徴量（float にキャスト） ===
        ord_arrays = []
        for col in cfg.ordinal_features:
            vals = df[col].to_numpy(dtype=np.float32)
            ord_arrays.append(vals)
        self.ord_features = torch.from_numpy(
            np.column_stack(ord_arrays) if ord_arrays else np.empty((len(df), 0), dtype=np.float32)
        )

        # === ターゲット ===
        self.swing_attempt = torch.from_numpy(df[cfg.target_cls_swing_attempt].to_numpy(dtype=np.float32))
        sr = df[cfg.target_cls_swing_result].to_numpy(dtype=np.int64, na_value=-1)
        self.swing_result = torch.from_numpy(sr)
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

    def _base_item(self, idx: int) -> dict[str, torch.Tensor]:
        """全サブクラス共通のサンプル辞書を構築する."""
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

    # --- Sequence / BatterHist 共有ヘルパー ---

    def _build_atbat_groups(self, at_bat_ids: np.ndarray) -> None:
        """at_bat_id によるグルーピングインデックスを構築する."""
        self.atbat_groups: list[np.ndarray] = []
        self.sample_to_group: np.ndarray = np.empty((len(at_bat_ids), 2), dtype=np.int64)

        current_id = at_bat_ids[0]
        group_start = 0
        group_idx = 0
        for row_idx in range(1, len(at_bat_ids)):
            if at_bat_ids[row_idx] != current_id:
                indices = np.arange(group_start, row_idx)
                self.atbat_groups.append(indices)
                for pos, ri in enumerate(indices):
                    self.sample_to_group[ri] = (group_idx, pos)
                group_idx += 1
                group_start = row_idx
                current_id = at_bat_ids[row_idx]
        # 最後のグループ
        indices = np.arange(group_start, len(at_bat_ids))
        self.atbat_groups.append(indices)
        for pos, ri in enumerate(indices):
            self.sample_to_group[ri] = (group_idx, pos)

    def _build_seq_features(self, past_indices: np.ndarray, max_seq_len: int) -> dict[str, torch.Tensor]:
        """打席内投球シーケンスの特徴量テンソルを構築する（右パディング）."""
        if len(past_indices) > max_seq_len:
            past_indices = past_indices[-max_seq_len:]
        seq_len = len(past_indices)
        T = max_seq_len

        seq_pitch_type = torch.zeros(T, dtype=torch.long)
        seq_cont = torch.zeros(T, self.num_cont, dtype=torch.float32)
        seq_swing_attempt = torch.zeros(T, dtype=torch.float32)
        seq_swing_result = torch.full((T,), -1, dtype=torch.long)
        seq_mask = torch.zeros(T, dtype=torch.float32)

        if seq_len > 0:
            pi = torch.from_numpy(past_indices.copy())
            seq_pitch_type[:seq_len] = self.cat_features["pitch_type"][pi]
            seq_cont[:seq_len] = self.cont_features[pi]
            seq_swing_attempt[:seq_len] = self.swing_attempt[pi]
            seq_swing_result[:seq_len] = self.swing_result[pi]
            seq_mask[:seq_len] = 1.0

        return {
            "seq_pitch_type": seq_pitch_type,
            "seq_cont": seq_cont,
            "seq_swing_attempt": seq_swing_attempt,
            "seq_swing_result": seq_swing_result,
            "seq_mask": seq_mask,
        }
