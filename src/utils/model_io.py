"""モデルの構築・保存・復元ユーティリティ."""

import dataclasses
import json
from pathlib import Path

import torch
import torch.nn as nn

from config import DataConfig, ModelConfig
from datasets import compute_embedding_dim, get_num_classes
from models import create_model


def build_model(data_cfg: DataConfig, model_cfg: ModelConfig, stats: dict) -> nn.Module:
    """stats 情報から embedding_dims を設定してモデルを構築する."""
    num_classes = get_num_classes(stats)

    # 入力カテゴリカル特徴量のカーディナリティ
    cat_cardinality = {
        "p_throws": num_classes.get("p_throws", 2),
        "pitch_type": num_classes.get("pitch_type", 18),
        "batter": num_classes.get("batter", 783),
        "stand": num_classes.get("stand", 2),
        "base_out_state": 24,
        "count_state": 12,
    }

    model_cfg.embedding_dims = {
        feat: (cat_cardinality[feat], compute_embedding_dim(cat_cardinality[feat]))
        for feat in data_cfg.categorical_features
    }

    model_cfg.num_swing_result = num_classes.get("swing_result", 3)
    model_cfg.num_bb_type = num_classes.get("bb_type", 4)

    num_cont = len(data_cfg.continuous_features)
    num_ord = len(data_cfg.ordinal_features)

    return create_model(model_cfg, num_cont, num_ord)


def save_model_config(model_cfg: ModelConfig, data_cfg: DataConfig, output_dir: Path) -> None:
    """モデル設定を model_config.json に保存する."""
    model_info = dataclasses.asdict(model_cfg)
    # embedding_dims のタプル値を list に変換（JSON 互換性）
    model_info["embedding_dims"] = {k: list(v) for k, v in model_cfg.embedding_dims.items()}
    # DataConfig 由来の追加情報
    model_info["num_cont"] = len(data_cfg.continuous_features)
    model_info["num_ord"] = len(data_cfg.ordinal_features)

    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_info, f, indent=2)


def load_trained_model(
    model_path: Path,
    model_config_path: Path,
    device: torch.device,
) -> nn.Module:
    """保存済みの重みとモデル設定からモデルを復元する."""
    with open(model_config_path) as f:
        saved = json.load(f)

    # embedding_dims の list → tuple 変換
    saved["embedding_dims"] = {k: tuple(v) for k, v in saved["embedding_dims"].items()}

    # ModelConfig のフィールド名を取得し、saved から該当するもののみ抽出
    cfg_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg_kwargs = {k: v for k, v in saved.items() if k in cfg_fields}
    model_cfg = ModelConfig(**cfg_kwargs)

    model = create_model(model_cfg, saved["num_cont"], saved["num_ord"])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
