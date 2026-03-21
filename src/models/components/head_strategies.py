"""Head 接続戦略コンポーネント."""

import torch
import torch.nn as nn

from config import ModelConfig
from models.components.heads import MDNHead, build_mlp_head


class IndependentHeadStrategy(nn.Module):
    """全ヘッドが backbone 出力を独立に受け取る."""

    def __init__(self, cfg: ModelConfig, backbone_out: int):
        super().__init__()
        activation = cfg.head_activation

        self.head_swing_attempt = build_mlp_head(backbone_out, cfg.head_hidden, 1, cfg.dropout, activation)
        self.head_swing_result = build_mlp_head(
            backbone_out, cfg.head_hidden, cfg.num_swing_result, cfg.dropout, activation
        )
        self.head_bb_type = build_mlp_head(backbone_out, cfg.head_hidden, cfg.num_bb_type, cfg.dropout, activation)

        if cfg.regression_head_type == "mdn":
            self.head_regression = MDNHead(
                in_dim=backbone_out,
                hidden_dims=cfg.head_hidden,
                out_dim=cfg.num_reg_targets,
                num_components=cfg.mdn_num_components,
                dropout=cfg.dropout,
            )
        else:
            self.head_regression = build_mlp_head(backbone_out, cfg.head_hidden, cfg.num_reg_targets, cfg.dropout, activation)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "swing_attempt": self.head_swing_attempt(h).squeeze(-1),
            "swing_result": self.head_swing_result(h),
            "bb_type": self.head_bb_type(h),
            "regression": self.head_regression(h),
        }


class CascadeHeadStrategy(nn.Module):
    """上流ヘッドの出力を下流ヘッドの入力に結合する.

    swing_attempt → swing_result → bb_type → regression
    """

    def __init__(self, cfg: ModelConfig, backbone_out: int):
        super().__init__()
        self.detach_cascade = cfg.detach_cascade
        activation = cfg.head_activation

        sa_out_dim = 1
        self.head_swing_attempt = build_mlp_head(backbone_out, cfg.head_hidden, sa_out_dim, cfg.dropout, activation)

        sr_in_dim = backbone_out + sa_out_dim
        self.head_swing_result = build_mlp_head(
            sr_in_dim, cfg.head_hidden, cfg.num_swing_result, cfg.dropout, activation
        )

        bt_in_dim = backbone_out + cfg.num_swing_result
        self.head_bb_type = build_mlp_head(bt_in_dim, cfg.head_hidden, cfg.num_bb_type, cfg.dropout, activation)

        reg_in_dim = backbone_out + cfg.num_bb_type
        if cfg.regression_head_type == "mdn":
            self.head_regression = MDNHead(
                in_dim=reg_in_dim,
                hidden_dims=cfg.head_hidden,
                out_dim=cfg.num_reg_targets,
                num_components=cfg.mdn_num_components,
                dropout=cfg.dropout,
            )
        else:
            self.head_regression = build_mlp_head(reg_in_dim, cfg.head_hidden, cfg.num_reg_targets, cfg.dropout, activation)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        detach = self.detach_cascade

        sa_logit = self.head_swing_attempt(h)
        sa_pass = sa_logit.detach() if detach else sa_logit

        sr_logit = self.head_swing_result(torch.cat([h, sa_pass], dim=-1))
        sr_pass = sr_logit.detach() if detach else sr_logit

        bt_logit = self.head_bb_type(torch.cat([h, sr_pass], dim=-1))
        bt_pass = bt_logit.detach() if detach else bt_logit

        reg_out = self.head_regression(torch.cat([h, bt_pass], dim=-1))

        return {
            "swing_attempt": sa_logit.squeeze(-1),
            "swing_result": sr_logit,
            "bb_type": bt_logit,
            "regression": reg_out,
        }


HEAD_STRATEGY_REGISTRY: dict[str, type[nn.Module]] = {
    "independent": IndependentHeadStrategy,
    "cascade": CascadeHeadStrategy,
}
