"""Feature Embedding コンポーネント."""

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """カテゴリカル特徴量の Embedding + 連続/序数特徴量の結合."""

    def __init__(self, embedding_dims: dict[str, tuple[int, int]]):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.embeddings = nn.ModuleDict()
        embed_total_dim = 0
        for feat_name, (num_classes, embed_dim) in embedding_dims.items():
            self.embeddings[feat_name] = nn.Embedding(num_classes + 1, embed_dim, padding_idx=num_classes)
            embed_total_dim += embed_dim
        self._embed_total_dim = embed_total_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_total_dim

    def forward(self, cat_dict: dict[str, torch.Tensor], cont: torch.Tensor, ord_feat: torch.Tensor) -> torch.Tensor:
        embeds = []
        for feat_name, (num_classes, _) in self.embedding_dims.items():
            x = cat_dict[feat_name]
            x = torch.where((x < 0) | (x >= num_classes), num_classes, x)
            embeds.append(self.embeddings[feat_name](x))
        return torch.cat(embeds + [cont, ord_feat], dim=-1)
