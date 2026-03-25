"""物理的整合性損失（PhysicsLoss）.

分類予測（bb_type, swing_result）と回帰予測（launch_angle, spray_angle）の間に
物理的一貫性を持たせるためのペナルティ損失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """分類予測と回帰予測の物理的整合性を強制するペナルティ損失.

    ソフト確率（softmax）で重み付けし、torch.relu ベースの微分可能ペナルティを計算する。
    回帰ターゲットが z-score 正規化されている前提で、物理閾値を正規化空間に変換して保持する。

    Args:
        reg_norm_stats: 回帰ターゲットの正規化パラメータ {col_name: (mean, std)}
        target_reg_columns: 回帰ターゲットのカラム名リスト（順序が出力テンソルのインデックスに対応）
        margin: 境界付近のマージン（度、生空間）。境界上の曖昧なサンプルへのペナルティを緩和する。
    """

    def __init__(
        self,
        reg_norm_stats: dict[str, tuple[float, float]],
        target_reg_columns: list[str],
        margin: float = 2.0,
    ) -> None:
        super().__init__()

        self.la_idx = target_reg_columns.index("launch_angle")
        self.sa_idx = target_reg_columns.index("spray_angle")

        la_mean, la_std = reg_norm_stats["launch_angle"]
        sa_mean, sa_std = reg_norm_stats["spray_angle"]

        def _norm(value: float, mean: float, std: float) -> torch.Tensor:
            return torch.tensor((value - mean) / std)

        # launch_angle 閾値（Statcast 基準: GB<10°, LD 10-25°, FB 25-50°, PU>50°）
        self.register_buffer("t_gb_ld", _norm(10.0, la_mean, la_std))
        self.register_buffer("t_ld_fb", _norm(25.0, la_mean, la_std))
        self.register_buffer("t_fb_pu", _norm(50.0, la_mean, la_std))

        # spray_angle 閾値（フェアゾーン: -45° 〜 +45°）
        self.register_buffer("t_sa_min", _norm(-45.0, sa_mean, sa_std))
        self.register_buffer("t_sa_max", _norm(45.0, sa_mean, sa_std))

        # マージンを正規化空間に変換
        self.register_buffer("margin_la", torch.tensor(margin / la_std))
        self.register_buffer("margin_sa", torch.tensor(margin / sa_std))

    def _get_reg_pred(self, reg_out: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """回帰予測値を取得する。MDN の場合は期待値 E[y] = sum(pi * mu) を算出."""
        if isinstance(reg_out, dict):
            return (reg_out["pi"].unsqueeze(-1) * reg_out["mu"]).sum(dim=1)
        return reg_out

    def _launch_angle_penalty(self, la_pred: torch.Tensor, bb_probs: torch.Tensor) -> torch.Tensor:
        """bb_type × launch_angle の整合性ペナルティ（確率加重二乗ペナルティ）."""
        m = self.margin_la
        # (lower, upper) per bb_type class: [0=gb, 1=fb, 2=ld, 3=popup]
        bounds = [
            (None,             self.t_gb_ld + m),   # ground_ball: LA < 10°
            (self.t_ld_fb - m, self.t_fb_pu + m),   # fly_ball:    25° < LA < 50°
            (self.t_gb_ld - m, self.t_ld_fb + m),   # line_drive:  10° < LA < 25°
            (self.t_fb_pu - m, None),                # popup:       LA > 50°
        ]
        loss = torch.zeros_like(la_pred)
        for cls, (lo, hi) in enumerate(bounds):
            penalty = torch.zeros_like(la_pred)
            if lo is not None:
                penalty = penalty + torch.relu(lo - la_pred)
            if hi is not None:
                penalty = penalty + torch.relu(la_pred - hi)
            loss = loss + bb_probs[:, cls] * penalty**2
        return loss.mean()

    def _spray_angle_penalty(self, sa_pred: torch.Tensor, sr_probs: torch.Tensor) -> torch.Tensor:
        """swing_result × spray_angle の整合性ペナルティ."""
        m = self.margin_sa
        # hit_into_play (cls 1): フェアゾーン外へのはみ出しペナルティ
        pen_hip = torch.relu((self.t_sa_min - m) - sa_pred) + torch.relu(sa_pred - (self.t_sa_max + m))
        # foul (cls 0): フェアゾーン内にいるとペナルティ（両端 relu の積 > 0 で内部判定）
        pen_foul = torch.relu(sa_pred - (self.t_sa_min + m)) * torch.relu((self.t_sa_max - m) - sa_pred)
        loss = sr_probs[:, 1] * pen_hip**2 + sr_probs[:, 0] * pen_foul
        return loss.mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """物理的整合性ペナルティを計算する."""
        device = outputs["bb_type"].device
        reg_pred = self._get_reg_pred(outputs["regression"])
        total = torch.tensor(0.0, device=device)

        def _valid_mask(cls_key: str, reg_col_idx: int) -> torch.Tensor:
            return (batch[cls_key] >= 0) & (batch["reg_mask"][:, reg_col_idx] > 0.5)

        # bb_type × launch_angle
        mask = _valid_mask("bb_type", self.la_idx)
        if mask.any():
            total = total + self._launch_angle_penalty(
                reg_pred[mask, self.la_idx], F.softmax(outputs["bb_type"][mask], dim=-1),
            )

        # swing_result × spray_angle
        if "swing_result" in outputs:
            mask = _valid_mask("swing_result", self.sa_idx)
            if mask.any():
                total = total + self._spray_angle_penalty(
                    reg_pred[mask, self.sa_idx], F.softmax(outputs["swing_result"][mask], dim=-1),
                )

        return total
