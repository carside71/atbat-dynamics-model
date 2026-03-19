"""カスケードヘッド付き残差 DNN モデル.

上流ヘッドの出力を下流ヘッドの入力に結合することで、
タスク間の階層的な依存関係をモデルに反映させる。

  swing_attempt → swing_result → bb_type → regression
"""

import torch
import torch.nn as nn

from config import ModelConfig
from models import register_model
from models.atbat_resdnn import ProjectedResBlock, ResBlock


@register_model("atbat_resdnn_cascade")
class AtBatResDNNCascade(nn.Module):
    """カスケードヘッド + 残差接続 + GELU による打席結果予測モデル.

    改良点（atbat_resdnn からの追加）:
      - ヘッド間の情報伝達: 上流ヘッドの出力を下流ヘッドの入力に結合
      - detach_cascade: True なら下流からの勾配が上流ヘッドに逆流しない（安定）

    階層構造:
      入力 → バックボーン → h
        h                           → Head1 → sa_logit
        [h, sa_logit]               → Head2 → sr_logit
        [h, sr_logit]               → Head3 → bt_logit
        [h, bt_logit]               → Head4 → reg_out
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

        # === Cascade heads ===
        # Head1: swing_attempt (入力: h)
        sa_out_dim = 1
        self.head_swing_attempt = self._build_head(backbone_out, cfg.head_hidden, sa_out_dim)

        # Head2: swing_result (入力: h + sa_logit)
        sr_in_dim = backbone_out + sa_out_dim
        self.head_swing_result = self._build_head(sr_in_dim, cfg.head_hidden, cfg.num_swing_result)

        # Head3: bb_type (入力: h + sr_logits)
        bt_in_dim = backbone_out + cfg.num_swing_result
        self.head_bb_type = self._build_head(bt_in_dim, cfg.head_hidden, cfg.num_bb_type)

        # Head4: regression (入力: h + bt_logits)
        reg_in_dim = backbone_out + cfg.num_bb_type
        self.head_regression = self._build_head(reg_in_dim, cfg.head_hidden, 3)

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
            x = x.clamp(min=0, max=num_classes)
            embeds.append(self.embeddings[feat_name](x))

        parts = embeds + [cont, ord_feat]
        x = torch.cat(parts, dim=-1)

        h = self.backbone(x)

        # Cascade: 上流の出力を下流の入力に結合
        detach = self.cfg.detach_cascade

        # Head1: swing_attempt
        sa_logit = self.head_swing_attempt(h)  # (B, 1)
        sa_pass = sa_logit.detach() if detach else sa_logit

        # Head2: swing_result (h + sa_logit)
        sr_logit = self.head_swing_result(torch.cat([h, sa_pass], dim=-1))
        sr_pass = sr_logit.detach() if detach else sr_logit

        # Head3: bb_type (h + sr_logits)
        bt_logit = self.head_bb_type(torch.cat([h, sr_pass], dim=-1))
        bt_pass = bt_logit.detach() if detach else bt_logit

        # Head4: regression (h + bt_logits)
        reg_out = self.head_regression(torch.cat([h, bt_pass], dim=-1))

        return {
            "swing_attempt": sa_logit.squeeze(-1),
            "swing_result": sr_logit,
            "bb_type": bt_logit,
            "regression": reg_out,
        }
