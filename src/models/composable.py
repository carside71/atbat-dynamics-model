"""Composable Model — コンポーネントを YAML 設定で組み立てるモデル."""

import torch
import torch.nn as nn

from config import ModelConfig
from models.components.backbones import BACKBONE_REGISTRY
from models.components.batter_hist_encoders import BATTER_HIST_ENCODER_REGISTRY
from models.components.embedding import FeatureEmbedding
from models.components.head_strategies import HEAD_STRATEGY_REGISTRY
from models.components.pitch_seq_encoders import PITCH_SEQ_ENCODER_REGISTRY


class ComposableModel(nn.Module):
    """YAML の model セクションから各コンポーネントを選択・組立するモデル."""

    def __init__(self, cfg: ModelConfig, num_cont: int, num_ord: int):
        super().__init__()
        self.cfg = cfg

        # 1. Feature Embedding
        self.embedding = FeatureEmbedding(cfg.embedding_dims)
        feat_dim = self.embedding.embed_dim + num_cont + num_ord

        # 2. 投球シーケンスエンコーダ（オプション）
        self.seq_encoder = None
        if cfg.pitch_seq_max_len > 0:
            encoder_cls = PITCH_SEQ_ENCODER_REGISTRY[cfg.pitch_seq_encoder_type]
            self.seq_encoder = encoder_cls(cfg, num_cont)
            feat_dim += self.seq_encoder.output_dim

        # 3. 打者履歴エンコーダ（オプション）
        self.hist_encoder = None
        if cfg.batter_hist_max_atbats > 0:
            hist_cls = BATTER_HIST_ENCODER_REGISTRY[cfg.batter_hist_encoder_type]
            self.hist_encoder = hist_cls(
                cfg,
                num_cont,
                seq_pitch_type_embed=self.seq_encoder.seq_pitch_type_embed if self.seq_encoder else None,
                seq_swing_result_embed=self.seq_encoder.seq_swing_result_embed if self.seq_encoder else None,
            )
            feat_dim += self.hist_encoder.output_dim

        # 4. Backbone
        backbone_cls = BACKBONE_REGISTRY[cfg.backbone_type]
        self.backbone = backbone_cls(feat_dim, cfg.backbone_hidden, cfg.dropout, cfg=cfg)

        # 5. Head Strategy
        strategy_cls = HEAD_STRATEGY_REGISTRY[cfg.head_strategy]
        self.head_strategy = strategy_cls(cfg, self.backbone.output_dim)

    @property
    def is_seq_model(self) -> bool:
        return self.seq_encoder is not None

    @property
    def is_batter_hist_model(self) -> bool:
        return self.hist_encoder is not None

    def forward(
        self,
        cat_dict: dict[str, torch.Tensor],
        cont: torch.Tensor,
        ord_feat: torch.Tensor,
        seq_pitch_type: torch.Tensor | None = None,
        seq_cont: torch.Tensor | None = None,
        seq_swing_attempt: torch.Tensor | None = None,
        seq_swing_result: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        hist_pitch_type: torch.Tensor | None = None,
        hist_cont: torch.Tensor | None = None,
        hist_swing_attempt: torch.Tensor | None = None,
        hist_swing_result: torch.Tensor | None = None,
        hist_bb_type: torch.Tensor | None = None,
        hist_launch_speed: torch.Tensor | None = None,
        hist_launch_angle: torch.Tensor | None = None,
        hist_spray_angle: torch.Tensor | None = None,
        hist_pitch_mask: torch.Tensor | None = None,
        hist_atbat_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B = cont.shape[0]
        device = cont.device

        # Embedding
        x = self.embedding(cat_dict, cont, ord_feat)

        # Sequence encoding
        if self.seq_encoder is not None:
            if seq_pitch_type is not None:
                seq_emb = self.seq_encoder(seq_pitch_type, seq_cont, seq_swing_attempt, seq_swing_result, seq_mask)
            else:
                seq_emb = torch.zeros(B, self.seq_encoder.output_dim, device=device)
            x = torch.cat([x, seq_emb], dim=-1)

        # Batter history encoding
        if self.hist_encoder is not None:
            if hist_pitch_type is not None:
                hist_emb = self.hist_encoder(
                    hist_pitch_type,
                    hist_cont,
                    hist_swing_attempt,
                    hist_swing_result,
                    hist_bb_type,
                    hist_launch_speed,
                    hist_launch_angle,
                    hist_spray_angle,
                    hist_pitch_mask,
                    hist_atbat_mask,
                )
            else:
                hist_emb = torch.zeros(B, self.hist_encoder.output_dim, device=device)
            x = torch.cat([x, hist_emb], dim=-1)

        # Backbone
        h = self.backbone(x)

        # Heads
        return self.head_strategy(h)
