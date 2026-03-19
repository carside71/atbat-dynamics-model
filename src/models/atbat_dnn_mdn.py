"""MDN (Mixture Density Network) ヘッドを持つマルチヘッド DNN モデル.

Head1-3 は AtBatDNN と同一。Head4 の回帰部分を混合ガウス分布の
パラメータ (π, μ, σ) を予測する MDN に置き換えたモデル。
"""

import torch
import torch.nn as nn

from config import ModelConfig
from models import register_model


class MDNHead(nn.Module):
    """Mixture Density Network ヘッド.

    K 個のガウス成分を用いて D 次元の出力を分布として予測する。
    出力: pi (B, K), mu (B, K, D), sigma (B, K, D)
    """

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, num_components: int, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.num_components = num_components

        # 共有隠れ層
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        self.shared = nn.Sequential(*layers)

        # 混合係数 π (K)
        self.fc_pi = nn.Linear(d, num_components)
        # 平均 μ (K * D)
        self.fc_mu = nn.Linear(d, num_components * out_dim)
        # 標準偏差 σ (K * D)  — softplus で正値化
        self.fc_sigma = nn.Linear(d, num_components * out_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(x)
        B = x.size(0)
        K, D = self.num_components, self.out_dim

        pi = torch.softmax(self.fc_pi(h), dim=-1)  # (B, K)
        mu = self.fc_mu(h).view(B, K, D)  # (B, K, D)
        sigma = nn.functional.softplus(self.fc_sigma(h)).view(B, K, D) + 1e-6  # (B, K, D)

        return {"pi": pi, "mu": mu, "sigma": sigma}


@register_model("atbat_dnn_mdn")
class AtBatDNNMDN(nn.Module):
    """打席結果予測の DNN + MDN 回帰ヘッド.

    分類ヘッド (Head1-3) は AtBatDNN と同一構造。
    回帰ヘッド (Head4) は MDN により混合ガウス分布を予測する。
    """

    def __init__(self, cfg: ModelConfig, num_cont: int, num_ord: int):
        super().__init__()
        self.cfg = cfg

        # === Embedding layers ===
        self.embeddings = nn.ModuleDict()
        embed_total_dim = 0
        for feat_name, (num_classes, embed_dim) in cfg.embedding_dims.items():
            self.embeddings[feat_name] = nn.Embedding(num_classes + 1, embed_dim, padding_idx=num_classes)
            embed_total_dim += embed_dim

        input_dim = embed_total_dim + num_cont + num_ord

        # === Shared backbone ===
        backbone_layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in cfg.backbone_hidden:
            backbone_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_layers)
        backbone_out = cfg.backbone_hidden[-1]

        # === Head: swing_attempt (binary) ===
        self.head_swing_attempt = self._build_head(backbone_out, cfg.head_hidden, 1)

        # === Head: swing_result (multi-class) ===
        self.head_swing_result = self._build_head(backbone_out, cfg.head_hidden, cfg.num_swing_result)

        # === Head: bb_type (multi-class) ===
        self.head_bb_type = self._build_head(backbone_out, cfg.head_hidden, cfg.num_bb_type)

        # === Head: regression (MDN) ===
        num_reg_targets = 3  # launch_speed, launch_angle, hit_distance_sc
        self.head_regression = MDNHead(
            in_dim=backbone_out,
            hidden_dims=cfg.head_hidden,
            out_dim=num_reg_targets,
            num_components=cfg.mdn_num_components,
            dropout=cfg.dropout,
        )

    def _build_head(self, in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.cfg.dropout)])
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
            x = torch.where((x < 0) | (x >= num_classes), num_classes, x)
            embeds.append(self.embeddings[feat_name](x))

        parts = embeds + [cont, ord_feat]
        x = torch.cat(parts, dim=-1)

        h = self.backbone(x)

        mdn_out = self.head_regression(h)

        return {
            "swing_attempt": self.head_swing_attempt(h).squeeze(-1),
            "swing_result": self.head_swing_result(h),
            "bb_type": self.head_bb_type(h),
            "regression": mdn_out,  # dict with pi, mu, sigma
        }
