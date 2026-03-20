"""モデルの構築・保存・復元ユーティリティ."""

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
    model_info = {
        "backbone_type": model_cfg.backbone_type,
        "embedding_dims": model_cfg.embedding_dims,
        "backbone_hidden": model_cfg.backbone_hidden,
        "head_hidden": model_cfg.head_hidden,
        "head_activation": model_cfg.head_activation,
        "head_strategy": model_cfg.head_strategy,
        "detach_cascade": model_cfg.detach_cascade,
        "regression_head_type": model_cfg.regression_head_type,
        "dropout": model_cfg.dropout,
        "num_swing_result": model_cfg.num_swing_result,
        "num_bb_type": model_cfg.num_bb_type,
        "mdn_num_components": model_cfg.mdn_num_components,
        "num_cont": len(data_cfg.continuous_features),
        "num_ord": len(data_cfg.ordinal_features),
        "max_seq_len": model_cfg.max_seq_len,
        "seq_encoder_type": model_cfg.seq_encoder_type,
        "seq_hidden_dim": model_cfg.seq_hidden_dim,
        "seq_num_layers": model_cfg.seq_num_layers,
        "seq_bidirectional": model_cfg.seq_bidirectional,
        "batter_hist_max_atbats": model_cfg.batter_hist_max_atbats,
        "batter_hist_max_pitches": model_cfg.batter_hist_max_pitches,
        "batter_hist_hidden_dim": model_cfg.batter_hist_hidden_dim,
        "batter_hist_num_layers": model_cfg.batter_hist_num_layers,
    }
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

    model_cfg = ModelConfig(
        backbone_type=saved.get("backbone_type", "resdnn"),
        embedding_dims={k: tuple(v) for k, v in saved["embedding_dims"].items()},
        backbone_hidden=saved["backbone_hidden"],
        head_hidden=saved["head_hidden"],
        head_activation=saved.get("head_activation", "gelu"),
        head_strategy=saved.get("head_strategy", "independent"),
        detach_cascade=saved.get("detach_cascade", True),
        regression_head_type=saved.get("regression_head_type", "mlp"),
        dropout=saved["dropout"],
        num_swing_result=saved["num_swing_result"],
        num_bb_type=saved["num_bb_type"],
        mdn_num_components=saved.get("mdn_num_components", 5),
        max_seq_len=saved.get("max_seq_len", 0),
        seq_encoder_type=saved.get("seq_encoder_type", "gru"),
        seq_hidden_dim=saved.get("seq_hidden_dim", 64),
        seq_num_layers=saved.get("seq_num_layers", 1),
        seq_bidirectional=saved.get("seq_bidirectional", False),
        batter_hist_max_atbats=saved.get("batter_hist_max_atbats", 0),
        batter_hist_max_pitches=saved.get("batter_hist_max_pitches", 10),
        batter_hist_hidden_dim=saved.get("batter_hist_hidden_dim", 64),
        batter_hist_num_layers=saved.get("batter_hist_num_layers", 1),
    )
    model = create_model(model_cfg, saved["num_cont"], saved["num_ord"])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
