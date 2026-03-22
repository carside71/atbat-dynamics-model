"""Statcast データセット."""

import torch

from datasets.statcast_base import StatcastBaseDataset


class StatcastDataset(StatcastBaseDataset):
    """Statcast データセット（単一投球サンプル）."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._base_item(idx)
