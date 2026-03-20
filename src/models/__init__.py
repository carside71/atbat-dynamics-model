"""モデルファクトリ."""

from __future__ import annotations

import torch.nn as nn

from config import ModelConfig


def create_model(cfg: ModelConfig, num_cont: int, num_ord: int) -> nn.Module:
    """ModelConfig からモデルを生成する."""
    from models.composable import ComposableModel

    return ComposableModel(cfg, num_cont, num_ord)
