"""シーケンス対応 Statcast データセット."""

import pandas as pd
import torch

from config import DataConfig
from datasets.statcast_base import StatcastBaseDataset


class StatcastSequenceDataset(StatcastBaseDataset):
    """シーケンス対応 Statcast データセット.

    同一 at_bat_id 内の過去投球を系列として提供する。
    データは投球順に並んでいることを前提とする。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DataConfig,
        max_seq_len: int = 10,
        norm_stats: dict[str, tuple[float, float]] | None = None,
        reg_norm_stats: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(df, cfg, norm_stats, reg_norm_stats)
        self.max_seq_len = max_seq_len
        self._build_atbat_groups(df["at_bat_id"].to_numpy())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group_idx, pos = int(self.sample_to_group[idx, 0]), int(self.sample_to_group[idx, 1])
        past_indices = self.atbat_groups[group_idx][:pos]

        result = self._base_item(idx)
        result.update(self._build_seq_features(past_indices, self.max_seq_len))
        return result
