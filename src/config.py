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

    # サンプルフィルタ設定 (regression スコープ用)
    filter_swing_attempt: bool = True   # swing_attempt==1 でフィルタするか
    reg_target_filter: str = "none"     # "none" | "any" | "all"

    # PCK 閾値（ターゲットごと、元スケール）
    pck_thresholds: dict[str, list[float]] = field(default_factory=lambda: {
        "launch_speed": [2.0, 5.0, 10.0],
        "launch_angle": [5.0, 10.0, 15.0],
        "hit_distance_sc": [10.0, 25.0, 50.0],
        "spray_angle": [5.0, 10.0, 15.0],
    })


_VALID_MODEL_SCOPES = {"all", "swing_attempt", "outcome", "classification", "regression"}


def validate_model_scope(scope: str) -> None:
    """model_scope の値を検証する."""
    if scope not in _VALID_MODEL_SCOPES:
        raise ValueError(f"Invalid model_scope={scope!r}. Must be one of {_VALID_MODEL_SCOPES}")


@dataclass
class HeatmapSubHeadConfig:
    """個別ヒートマップサブヘッドの設定."""

    type: str              # "1d" | "2d"
    targets: list[str]     # 1d: 1個、2d: 2個のターゲット名
    grid_h: int | None = None   # 2d のみ。None でグローバル値使用
    grid_w: int | None = None   # 2d のみ
    num_bins: int | None = None  # 1d のみ。None でグローバル値使用


@dataclass
class ModelConfig:
    # モデルスコープ: "all"=全タスク統合, "swing_attempt"=SA予測のみ, "outcome"=SR/BT/Reg予測のみ,
    # "classification"=SA/SR/BT分類のみ, "regression"=回帰のみ
    model_scope: str = "all"

    # カテゴリカル特徴量の埋め込み次元（実行時に stats から自動設定）
    embedding_dims: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Backbone
    backbone_type: str = "resdnn"  # "dnn" | "resdnn" | "attention"
    backbone_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.2

    # Self-Attention Backbone (backbone_type == "attention" のみ有効)
    attn_num_heads: int = 4           # マルチヘッド注意のヘッド数
    attn_num_layers: int = 2          # Transformer Encoder のレイヤー数
    attn_token_dim: int = 64          # トークン次元 (d_model)
    attn_ff_multiplier: int = 4       # FFN次元 = attn_token_dim * attn_ff_multiplier
    attn_pool: str = "cls"            # "cls" | "mean" — プーリング方式

    # Head Strategy
    head_strategy: str = "independent"  # "independent" | "cascade"
    head_hidden: list[int] = field(default_factory=lambda: [64])
    head_activation: str = "gelu"  # "relu" | "gelu"
    detach_cascade: bool = True  # cascade 時のみ有効

    # Regression Head
    regression_head_type: str = "mlp"  # "mlp" | "mdn" | "heatmap"
    mdn_num_components: int = 5  # mdn 時のみ有効
    num_reg_targets: int = 3  # target_reg の数（実行時に自動設定）

    # Heatmap Head 設定（regression_head_type == "heatmap" 時のみ有効）
    heatmap_grid_h: int = 64           # 2Dヒートマップの高さ (launch_angle 軸)
    heatmap_grid_w: int = 64           # 2Dヒートマップの幅 (spray_angle 軸)
    heatmap_num_bins: int = 64         # 1Dヒートマップのビン数
    # 物理値域（YAMLで指定）
    heatmap_range_launch_speed: list[float] = field(default_factory=lambda: [40.0, 120.0])   # mph
    heatmap_range_launch_angle: list[float] = field(default_factory=lambda: [-90.0, 90.0])   # deg
    heatmap_range_hit_distance: list[float] = field(default_factory=lambda: [0.0, 500.0])    # ft
    heatmap_range_spray_angle: list[float] = field(default_factory=lambda: [-45.0, 45.0])    # deg
    # 正規化値域（学習時に自動計算、推論時は model_config.json から読み込み）
    heatmap_norm_range_launch_speed: list[float] = field(default_factory=lambda: [-4.0, 4.0])
    heatmap_norm_range_launch_angle: list[float] = field(default_factory=lambda: [-4.0, 4.0])
    heatmap_norm_range_hit_distance: list[float] = field(default_factory=lambda: [-4.0, 4.0])
    heatmap_norm_range_spray_angle: list[float] = field(default_factory=lambda: [-4.0, 4.0])
    heatmap_sigma: float = 2.0        # GTガウスの sigma（ピクセル単位）
    heatmap_intermediate_dim: int = 256  # deconv 前の中間チャネル数

    # 設定可能なヒートマップヘッド構成（None = レガシーハードコード動作）
    # YAML 例:
    #   heatmap_heads:
    #     - type: "2d"
    #       targets: [launch_angle, spray_angle]
    #     - type: "1d"
    #       targets: [launch_speed]
    #     - type: "1d"
    #       targets: [hit_distance_sc]
    heatmap_heads: list[dict] | None = None
    # 学習時に自動設定: target_reg のターゲット名リスト
    heatmap_target_reg: list[str] | None = None
    # 学習時に自動設定: ターゲット名 → 正規化値域の dict
    heatmap_norm_ranges: dict[str, list[float]] | None = None

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

    def get_heatmap_head_configs(self) -> list[HeatmapSubHeadConfig] | None:
        """heatmap_heads を HeatmapSubHeadConfig のリストに変換する.

        Returns:
            None の場合はレガシーモード（ハードコード動作）。
        """
        if self.heatmap_heads is None:
            return None
        configs = [HeatmapSubHeadConfig(**h) for h in self.heatmap_heads]
        # バリデーション
        for hc in configs:
            if hc.type == "2d" and len(hc.targets) != 2:
                raise ValueError(f"2D heatmap head must have exactly 2 targets, got {hc.targets}")
            if hc.type == "1d" and len(hc.targets) != 1:
                raise ValueError(f"1D heatmap head must have exactly 1 target, got {hc.targets}")
            if hc.type not in ("1d", "2d"):
                raise ValueError(f"Invalid heatmap head type: {hc.type!r}")
        # 重複チェック
        all_targets = [t for hc in configs for t in hc.targets]
        if len(all_targets) != len(set(all_targets)):
            raise ValueError(f"Duplicate targets in heatmap_heads: {all_targets}")
        return configs

    def get_heatmap_norm_range(self, target: str) -> tuple[float, float]:
        """指定ターゲットの正規化値域を取得する（新 dict → 旧個別フィールド fallback）."""
        if self.heatmap_norm_ranges is not None and target in self.heatmap_norm_ranges:
            r = self.heatmap_norm_ranges[target]
            return (r[0], r[1])
        # 旧個別フィールドへ fallback
        _legacy_map = {
            "launch_speed": self.heatmap_norm_range_launch_speed,
            "launch_angle": self.heatmap_norm_range_launch_angle,
            "hit_distance_sc": self.heatmap_norm_range_hit_distance,
            "spray_angle": self.heatmap_norm_range_spray_angle,
        }
        if target in _legacy_map:
            r = _legacy_map[target]
            return (r[0], r[1])
        raise ValueError(f"No heatmap norm range found for target: {target!r}")


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

    # Heatmap Loss 設定
    heatmap_loss_weight_offset: float = 1.0  # オフセット損失の重み（heatmap focal loss に対する比率）
    heatmap_focal_alpha: float = 2.0  # focal loss の alpha パラメータ
    heatmap_focal_beta: float = 4.0   # focal loss の beta パラメータ


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
