"""残差接続 + GELU を用いたマルチヘッド DNN モデル."""

import torch
import torch.nn as nn

from config import ModelConfig
from models import register_model


class ResBlock(nn.Module):
    """残差ブロック: Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm + Skip."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ProjectedResBlock(nn.Module):
    """次元が変わる場合の残差ブロック: 射影ショートカット付き."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x) + self.net(x))


@register_model("atbat_resdnn")
class AtBatResDNN(nn.Module):
    """残差接続 + GELU による打席結果予測モデル.

    改良点:
      - ReLU → GELU: 滑らかな活性化でテーブルデータの表現力を向上
      - 残差接続: 勾配伝搬を改善し、より深いネットワークを安定して学習

    階層構造:
      入力 → 共有バックボーン(ResBlock) → Head1: swing_attempt (binary)
                                        → Head2: swing_result (9-class)
                                        → Head3: bb_type (4-class)
                                        → Head4: regression (launch_speed, launch_angle, hit_distance_sc)
    """

    def __init__(self, cfg: ModelConfig, num_cont: int, num_ord: int):
        super().__init__()
        self.cfg = cfg

        # === Embedding layers ===
        self.embeddings = nn.ModuleDict()
        embed_total_dim = 0
        for feat_name, (num_classes, embed_dim) in cfg.embedding_dims.items():
            # +1 for unknown/padding index
            self.embeddings[feat_name] = nn.Embedding(num_classes + 1, embed_dim, padding_idx=num_classes)
            embed_total_dim += embed_dim

        input_dim = embed_total_dim + num_cont + num_ord

        # === Shared backbone (ResBlocks) ===
        blocks: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in cfg.backbone_hidden:
            if in_dim != hidden_dim:
                blocks.append(ProjectedResBlock(in_dim, hidden_dim, cfg.dropout))
            else:
                blocks.append(ResBlock(hidden_dim, cfg.dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        backbone_out = cfg.backbone_hidden[-1]

        # === Head: swing_attempt (binary) ===
        self.head_swing_attempt = self._build_head(backbone_out, cfg.head_hidden, 1)

        # === Head: swing_result (multi-class) ===
        self.head_swing_result = self._build_head(backbone_out, cfg.head_hidden, cfg.num_swing_result)

        # === Head: bb_type (multi-class) ===
        self.head_bb_type = self._build_head(backbone_out, cfg.head_hidden, cfg.num_bb_type)

        # === Head: regression (launch_speed, launch_angle, hit_distance_sc) ===
        self.head_regression = self._build_head(backbone_out, cfg.head_hidden, 3)

    def _build_head(self, in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(d, h),
                    nn.GELU(),
                    nn.Dropout(self.cfg.dropout),
                ]
            )
            d = h
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    def forward(
        self, cat_dict: dict[str, torch.Tensor], cont: torch.Tensor, ord_feat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Embed categorical features
        embeds = []
        for feat_name in self.cfg.embedding_dims:
            x = cat_dict[feat_name]
            num_classes = self.cfg.embedding_dims[feat_name][0]
            # 不正値(-1やrange外)をpadding_idxにマッピング
            x = x.clamp(min=0, max=num_classes)
            embeds.append(self.embeddings[feat_name](x))

        # Concatenate all input features
        parts = embeds + [cont, ord_feat]
        x = torch.cat(parts, dim=-1)

        # Shared backbone (ResBlocks)
        h = self.backbone(x)

        return {
            "swing_attempt": self.head_swing_attempt(h).squeeze(-1),
            "swing_result": self.head_swing_result(h),
            "bb_type": self.head_bb_type(h),
            "regression": self.head_regression(h),
        }
