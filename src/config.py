"""モデル・学習の設定."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_dir: Path = Path("/workspace/datasets/statcast-customized-v2")
    output_dir: Path = Path("/workspace/outputs/atbat-dynamics-model")

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
            "vx0",
            "vy0",
            "vz0",
            "ax",
            "ay",
            "az",
            "sz_top",
            "sz_bot",
            "plate_z_norm",
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

    # 後方互換プロパティ
    @property
    def data_dir(self) -> Path:
        return self.dataset_dir

    @property
    def stats_dir(self) -> Path:
        return self.dataset_dir

    @property
    def split_dir(self) -> Path:
        return self.dataset_dir

    @property
    def batter_history_dir(self) -> Path:
        return self.dataset_dir

    # ターゲット
    target_cls_swing_attempt: str = "swing_attempt"
    target_cls_swing_result: str = "swing_result"
    target_cls_bb_type: str = "bb_type"
    target_reg: list[str] = field(
        default_factory=lambda: [
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
            "spray_angle",
        ]
    )


_VALID_MODEL_SCOPES = {"all", "swing_attempt", "outcome", "classification", "regression"}


def validate_model_scope(scope: str) -> None:
    """model_scope の値を検証する."""
    if scope not in _VALID_MODEL_SCOPES:
        raise ValueError(f"Invalid model_scope={scope!r}. Must be one of {_VALID_MODEL_SCOPES}")


@dataclass
class ModelConfig:
    # モデルスコープ: "all"=全タスク統合, "swing_attempt"=SA予測のみ, "outcome"=SR/BT/Reg予測のみ,
    # "classification"=SA/SR/BT分類のみ, "regression"=回帰のみ
    model_scope: str = "all"

    # カテゴリカル特徴量の埋め込み次元（実行時に stats から自動設定）
    embedding_dims: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Backbone
    backbone_type: str = "resdnn"  # "dnn" | "resdnn"
    backbone_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.2

    # Head Strategy
    head_strategy: str = "independent"  # "independent" | "cascade"
    head_hidden: list[int] = field(default_factory=lambda: [64])
    head_activation: str = "gelu"  # "relu" | "gelu"
    detach_cascade: bool = True  # cascade 時のみ有効

    # Regression Head
    regression_head_type: str = "mlp"  # "mlp" | "mdn"
    mdn_num_components: int = 5  # mdn 時のみ有効
    num_reg_targets: int = 3  # target_reg の数（実行時に自動設定）

    # 出力クラス数（実行時に自動設定）
    num_swing_result: int = 3
    num_bb_type: int = 4

    # 投球シーケンスエンコーダ（0 で無効）
    pitch_seq_max_len: int = 0
    pitch_seq_encoder_type: str = "gru"  # "gru" | "transformer"
    pitch_seq_hidden_dim: int = 64
    pitch_seq_num_layers: int = 1
    pitch_seq_bidirectional: bool = False

    # 打者履歴エンコーダ（0 で無効）
    batter_hist_max_atbats: int = 0
    batter_hist_max_pitches: int = 10
    batter_hist_encoder_type: str = "gru"  # "gru" | "transformer"
    batter_hist_hidden_dim: int = 64
    batter_hist_num_layers: int = 1


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

    # Focal Loss 設定
    focal_gamma: float = 0.0  # 0.0 で通常の cross-entropy と同等
    use_class_weight: bool = False  # クラス頻度の逆数で重み付け

    # Label Smoothing
    label_smoothing: float = 0.0  # 0.0 で無効、0.1 程度が一般的

    # Physics Consistency Loss
    loss_weight_physics: float = 0.0  # 0.0 で無効（後方互換）
    physics_margin_degrees: float = 2.0  # 境界付近のマージン（度）


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
