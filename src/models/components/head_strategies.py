"""Head 接続戦略コンポーネント."""

import torch
import torch.nn as nn

from config import ModelConfig
from models.components.heads import MDNHead, build_mlp_head


def _build_regression_head(cfg: ModelConfig, in_dim: int) -> nn.Module:
    """回帰ヘッドを構築する（MLP / MDN / Heatmap）."""
    if cfg.regression_head_type == "heatmap":
        from models.components.heatmap_head import HeatmapHead

        return HeatmapHead(in_dim=in_dim, cfg=cfg)
    if cfg.regression_head_type == "mdn":
        return MDNHead(
            in_dim=in_dim,
            hidden_dims=cfg.head_hidden,
            out_dim=cfg.num_reg_targets,
            num_components=cfg.mdn_num_components,
            dropout=cfg.dropout,
        )
    return build_mlp_head(in_dim, cfg.head_hidden, cfg.num_reg_targets, cfg.dropout, cfg.head_activation)


class IndependentHeadStrategy(nn.Module):
    """全ヘッドが backbone 出力を独立に受け取る."""

    def __init__(self, cfg: ModelConfig, backbone_out: int):
        super().__init__()
        self.model_scope = cfg.model_scope
        activation = cfg.head_activation

        if self.model_scope in ("all", "swing_attempt", "classification"):
            self.head_swing_attempt = build_mlp_head(backbone_out, cfg.head_hidden, 1, cfg.dropout, activation)

        if self.model_scope in ("all", "outcome", "classification"):
            self.head_swing_result = build_mlp_head(
                backbone_out, cfg.head_hidden, cfg.num_swing_result, cfg.dropout, activation
            )
            self.head_bb_type = build_mlp_head(backbone_out, cfg.head_hidden, cfg.num_bb_type, cfg.dropout, activation)

        if self.model_scope in ("all", "outcome", "regression"):
            self.head_regression = _build_regression_head(cfg, backbone_out)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.model_scope in ("all", "swing_attempt", "classification"):
            out["swing_attempt"] = self.head_swing_attempt(h).squeeze(-1)
        if self.model_scope in ("all", "outcome", "classification"):
            out["swing_result"] = self.head_swing_result(h)
            out["bb_type"] = self.head_bb_type(h)
        if self.model_scope in ("all", "outcome", "regression"):
            out["regression"] = self.head_regression(h)
        return out


class CascadeHeadStrategy(nn.Module):
    """上流ヘッドの出力を下流ヘッドの入力に結合する.

    swing_attempt → swing_result → bb_type → regression
    """

    def __init__(self, cfg: ModelConfig, backbone_out: int):
        super().__init__()
        self.model_scope = cfg.model_scope
        self.detach_cascade = cfg.detach_cascade
        activation = cfg.head_activation

        if self.model_scope in ("all", "swing_attempt", "classification"):
            sa_out_dim = 1
            self.head_swing_attempt = build_mlp_head(backbone_out, cfg.head_hidden, sa_out_dim, cfg.dropout, activation)

        if self.model_scope in ("all", "outcome", "classification"):
            # SA logit をカスケード入力に含めるかどうか
            has_sa = self.model_scope in ("all", "classification")
            sr_in_dim = (backbone_out + 1) if has_sa else backbone_out
            self.head_swing_result = build_mlp_head(
                sr_in_dim, cfg.head_hidden, cfg.num_swing_result, cfg.dropout, activation
            )

            bt_in_dim = backbone_out + cfg.num_swing_result
            self.head_bb_type = build_mlp_head(bt_in_dim, cfg.head_hidden, cfg.num_bb_type, cfg.dropout, activation)

        if self.model_scope in ("all", "outcome", "regression"):
            # regression の入力: 分類ヘッドがある場合は bb_type logit を結合、なければ backbone のみ
            has_cls = self.model_scope in ("all", "outcome")
            reg_in_dim = (backbone_out + cfg.num_bb_type) if has_cls else backbone_out
            self.head_regression = _build_regression_head(cfg, reg_in_dim)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        detach = self.detach_cascade

        if self.model_scope in ("all", "swing_attempt", "classification"):
            sa_logit = self.head_swing_attempt(h)
            out["swing_attempt"] = sa_logit.squeeze(-1)

        if self.model_scope in ("all", "outcome", "classification"):
            if self.model_scope in ("all", "classification"):
                sa_pass = sa_logit.detach() if detach else sa_logit
                sr_input = torch.cat([h, sa_pass], dim=-1)
            else:
                sr_input = h

            sr_logit = self.head_swing_result(sr_input)
            sr_pass = sr_logit.detach() if detach else sr_logit

            bt_logit = self.head_bb_type(torch.cat([h, sr_pass], dim=-1))

            out["swing_result"] = sr_logit
            out["bb_type"] = bt_logit

        if self.model_scope in ("all", "outcome", "regression"):
            if self.model_scope in ("all", "outcome"):
                bt_pass = bt_logit.detach() if detach else bt_logit
                reg_out = self.head_regression(torch.cat([h, bt_pass], dim=-1))
            else:
                reg_out = self.head_regression(h)
            out["regression"] = reg_out

        return out


HEAD_STRATEGY_REGISTRY: dict[str, type[nn.Module]] = {
    "independent": IndependentHeadStrategy,
    "cascade": CascadeHeadStrategy,
}
