"""Backbone コンポーネント."""

import torch
import torch.nn as nn

from utils.registry import make_registry

BACKBONE_REGISTRY, register_backbone = make_registry()


@register_backbone("dnn")
class DNNBackbone(nn.Module):
    """Linear → BatchNorm → ReLU → Dropout の繰返し."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
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

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
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
