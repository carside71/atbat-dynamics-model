"""Focal Loss の実装.

クラス不均衡に対処するため、簡単なサンプルの損失を抑制し、
分類が困難なサンプルに注目する。

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """マルチクラス分類向け Focal Loss.

    Args:
        gamma: 焦点パラメータ。大きいほど簡単なサンプルの重みが下がる（デフォルト: 2.0）。
        weight: クラスごとの重みテンソル（オプション）。
        reduction: 'mean' | 'sum' | 'none'。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss を計算する.

        Args:
            logits: (N, C) 未正規化ロジット
            targets: (N,) クラスインデックス
        """
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # 正解クラスの予測確率
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
