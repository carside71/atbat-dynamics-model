"""モデル・学習の設定."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: Path = Path("/workspace/datasets/statcast-customized/data")
    stats_dir: Path = Path("/workspace/datasets/statcast-customized/stats")
    output_dir: Path = Path("/workspace/outputs/atbat-dynamics-model")
    train_years: list[int] = field(default_factory=lambda: [2017, 2018, 2019, 2020, 2021, 2022, 2023])
    val_years: list[int] = field(default_factory=lambda: [2024])
    test_years: list[int] = field(default_factory=lambda: [2025])

    # 入力特徴量
    categorical_features: list[str] = field(
        default_factory=lambda: [
            "p_throws",
            "pitch_type",
            "batter",
            "stand",
            "base_out_state",
            "count_state",
        ]
    )
    continuous_features: list[str] = field(
        default_factory=lambda: [
            "release_speed",
            "release_spin_rate",
            "pfx_x",
            "pfx_z",
            "plate_x",
            "plate_z",
        ]
    )
    ordinal_features: list[str] = field(
        default_factory=lambda: [
            "inning_clipped",
            "is_inning_top",
            "diff_score_clipped",
            "pitch_number_clipped",
        ]
    )

    # ターゲット
    target_cls_swing_attempt: str = "swing_attempt"
    target_cls_swing_result: str = "swing_result"
    target_cls_bb_type: str = "bb_type"
    target_reg: list[str] = field(
        default_factory=lambda: [
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
        ]
    )


@dataclass
class ModelConfig:
    # モデルアーキテクチャ名（モデルレジストリのキー）
    architecture: str = "atbat_dnn"

    # カテゴリカル特徴量の埋め込み次元（num_classes → embed_dim のマッピング）
    # 実行時に stats ファイルから自動設定される
    embedding_dims: dict[str, tuple[int, int]] = field(default_factory=dict)

    # ネットワーク構造
    backbone_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    head_hidden: list[int] = field(default_factory=lambda: [64])
    dropout: float = 0.2

    # 出力クラス数
    num_swing_result: int = 9
    num_bb_type: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 4096
    num_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42

    # 損失の重み
    loss_weight_swing_attempt: float = 1.0
    loss_weight_swing_result: float = 1.0
    loss_weight_bb_type: float = 1.0
    loss_weight_regression: float = 0.01


def _apply_overrides(obj: Any, overrides: dict) -> None:
    """辞書の値をdataclassフィールドに適用する."""
    from dataclasses import fields as dc_fields

    field_annotations = {f.name: f.type for f in dc_fields(obj)}
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {key}")
        annotation = field_annotations.get(key)
        if annotation is Path or annotation == "Path":
            value = Path(value)
        setattr(obj, key, value)


def load_config(yaml_path: str | Path) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    """YAMLファイルから設定を読み込む.

    YAMLファイルには data, model, train の3つのセクションを記述できる。
    指定されていないフィールドはデフォルト値が使われる。
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if "data" in raw:
        _apply_overrides(data_cfg, raw["data"])
    if "model" in raw:
        _apply_overrides(model_cfg, raw["model"])
    if "train" in raw:
        _apply_overrides(train_cfg, raw["train"])

    return data_cfg, model_cfg, train_cfg
