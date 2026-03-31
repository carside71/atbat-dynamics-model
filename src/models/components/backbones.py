"""Backbone コンポーネント."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.registry import make_registry

BACKBONE_REGISTRY, register_backbone = make_registry()


@register_backbone("dnn")
class DNNBackbone(nn.Module):
    """Linear → BatchNorm → ReLU → Dropout の繰返し."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, **kwargs):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.net = nn.Sequential(*layers)
        self._output_dim = hidden_dims[-1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


@register_backbone("resdnn")
class ResDNNBackbone(nn.Module):
    """ResBlock / ProjectedResBlock の繰返し."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, **kwargs):
        super().__init__()
        blocks: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            if in_dim != h:
                blocks.append(ProjectedResBlock(in_dim, h, dropout))
            else:
                blocks.append(ResBlock(h, dropout))
            in_dim = h
        self.net = nn.Sequential(*blocks)
        self._output_dim = hidden_dims[-1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_backbone("attention")
class SelfAttentionBackbone(nn.Module):
    """フラット特徴量をトークン化し Self-Attention で処理するバックボーン (FT-Transformer 風)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        *,
        cfg=None,
    ):
        super().__init__()
        from config import ModelConfig

        if cfg is None:
            cfg = ModelConfig()

        token_dim = cfg.attn_token_dim
        num_heads = cfg.attn_num_heads
        num_layers = cfg.attn_num_layers
        ff_dim = token_dim * cfg.attn_ff_multiplier
        self.pool_mode = cfg.attn_pool
        self.num_tokens = max(1, math.ceil(input_dim / token_dim))

        # 入力をトークン列に射影
        self.tokenizer = nn.Linear(input_dim, self.num_tokens * token_dim)
        self.token_dim = token_dim

        # 学習可能位置埋め込み
        total_tokens = self.num_tokens + (1 if self.pool_mode == "cls" else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, token_dim) * 0.02)

        # CLS トークン
        if self.pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # 出力射影
        self.out_norm = nn.LayerNorm(token_dim)
        self.out_proj = nn.Linear(token_dim, hidden_dims[-1])
        self._output_dim = hidden_dims[-1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # トークン化: (B, input_dim) → (B, num_tokens, token_dim)
        tokens = self.tokenizer(x).view(B, self.num_tokens, self.token_dim)

        # CLS トークンを先頭に追加
        if self.pool_mode == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # 位置埋め込み
        tokens = tokens + self.pos_embed

        # Transformer Encoder
        tokens = self.encoder(tokens)

        # プーリング
        if self.pool_mode == "cls":
            pooled = tokens[:, 0]
        else:
            pooled = tokens.mean(dim=1)

        # 出力射影
        return self.out_proj(self.out_norm(pooled))
