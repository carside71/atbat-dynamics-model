"""テスト用共通 fixture."""

import sys
from pathlib import Path

import pytest
import torch

# src ディレクトリをインポートパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import ModelConfig, TrainConfig


@pytest.fixture()
def small_model_cfg():
    """テスト用の小さい ModelConfig を生成する."""
    return ModelConfig(
        model_scope="all",
        embedding_dims={
            "p_throws": (3, 2),
            "pitch_type": (5, 4),
        },
        backbone_type="dnn",
        backbone_hidden=[16, 8],
        head_hidden=[8],
        head_activation="relu",
        dropout=0.0,
        regression_head_type="mlp",
        num_swing_result=3,
        num_bb_type=4,
        num_reg_targets=5,
    )


@pytest.fixture()
def train_cfg():
    """テスト用の TrainConfig を生成する."""
    return TrainConfig(
        loss_weight_swing_attempt=1.0,
        loss_weight_swing_result=1.0,
        loss_weight_bb_type=1.0,
        loss_weight_regression=0.01,
    )


@pytest.fixture()
def fake_batch():
    """正しい shape/dtype のランダムバッチを生成する."""
    B = 8
    return {
        "p_throws": torch.randint(0, 3, (B,)),
        "pitch_type": torch.randint(0, 5, (B,)),
        "cont": torch.randn(B, 15),
        "ord": torch.randn(B, 4),
        "swing_attempt": torch.randint(0, 2, (B,)).float(),
        "swing_result": torch.tensor([0, 1, 2, -1, 0, 1, -1, 2]),
        "bb_type": torch.tensor([-1, 0, -1, -1, -1, 1, -1, -1]),
        "reg_targets": torch.randn(B, 5),
        "reg_mask": torch.tensor([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32),
    }
