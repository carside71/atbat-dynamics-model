"""物理的整合性損失（PhysicsConsistencyLoss）.

分類予測（bb_type, swing_result）と回帰予測（launch_angle, spray_angle）の間に
物理的一貫性を持たせるためのペナルティ損失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Statcast 基準の打球タイプ別 launch_angle 境界（度）
_LA_BOUNDARY_GB_LD = 10.0  # ground_ball / line_drive 境界
_LA_BOUNDARY_LD_FB = 25.0  # line_drive / fly_ball 境界
_LA_BOUNDARY_FB_PU = 50.0  # fly_ball / popup 境界

# フェアゾーンの spray_angle 境界（度）
_SA_FAIR_MIN = -45.0
_SA_FAIR_MAX = 45.0

# bb_type クラスインデックス
_CLS_GROUND_BALL = 0
_CLS_FLY_BALL = 1
_CLS_LINE_DRIVE = 2
_CLS_POPUP = 3

# swing_result クラスインデックス
_CLS_FOUL = 0
_CLS_HIT_INTO_PLAY = 1


def _normalize_threshold(value: float, mean: float, std: float) -> float:
    """生の角度値を正規化空間の値に変換する."""
    return (value - mean) / std


class PhysicsConsistencyLoss(nn.Module):
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

        # launch_angle 閾値を正規化空間に変換
        self.register_buffer(
            "t_gb_ld", torch.tensor(_normalize_threshold(_LA_BOUNDARY_GB_LD, la_mean, la_std))
        )
        self.register_buffer(
            "t_ld_fb", torch.tensor(_normalize_threshold(_LA_BOUNDARY_LD_FB, la_mean, la_std))
        )
        self.register_buffer(
            "t_fb_pu", torch.tensor(_normalize_threshold(_LA_BOUNDARY_FB_PU, la_mean, la_std))
        )

        # spray_angle 閾値を正規化空間に変換
        self.register_buffer(
            "t_sa_min", torch.tensor(_normalize_threshold(_SA_FAIR_MIN, sa_mean, sa_std))
        )
        self.register_buffer(
            "t_sa_max", torch.tensor(_normalize_threshold(_SA_FAIR_MAX, sa_mean, sa_std))
        )

        # マージンを正規化空間に変換
        self.register_buffer("margin_la", torch.tensor(margin / la_std))
        self.register_buffer("margin_sa", torch.tensor(margin / sa_std))

    def _get_reg_pred(self, reg_out: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """回帰予測値を取得する。MDN の場合は期待値を算出."""
        if isinstance(reg_out, dict):
            # MDN: E[y] = Σ_k π_k * μ_k
            pi = reg_out["pi"]  # (B, K)
            mu = reg_out["mu"]  # (B, K, D)
            return (pi.unsqueeze(-1) * mu).sum(dim=1)  # (B, D)
        return reg_out  # (B, D)

    def _launch_angle_penalty(
        self,
        la_pred: torch.Tensor,
        bb_probs: torch.Tensor,
    ) -> torch.Tensor:
        """bb_type × launch_angle の整合性ペナルティを計算する.

        Args:
            la_pred: (N,) 正規化空間の launch_angle 予測
            bb_probs: (N, 4) bb_type のソフト確率
        """
        m = self.margin_la

        # ground_ball (cls 0): LA < 10° → LA > t + m でペナルティ
        penalty_gb = torch.relu(la_pred - (self.t_gb_ld + m))

        # line_drive (cls 2): 10° ≤ LA ≤ 25° → 範囲外でペナルティ
        penalty_ld = torch.relu((self.t_gb_ld - m) - la_pred) + torch.relu(la_pred - (self.t_ld_fb + m))

        # fly_ball (cls 1): 25° < LA ≤ 50° → 範囲外でペナルティ
        penalty_fb = torch.relu((self.t_ld_fb - m) - la_pred) + torch.relu(la_pred - (self.t_fb_pu + m))

        # popup (cls 3): LA > 50° → LA < t - m でペナルティ
        penalty_pu = torch.relu((self.t_fb_pu - m) - la_pred)

        # 確率加重の二乗ペナルティ
        loss = (
            bb_probs[:, _CLS_GROUND_BALL] * penalty_gb**2
            + bb_probs[:, _CLS_LINE_DRIVE] * penalty_ld**2
            + bb_probs[:, _CLS_FLY_BALL] * penalty_fb**2
            + bb_probs[:, _CLS_POPUP] * penalty_pu**2
        )
        return loss.mean()

    def _spray_angle_penalty(
        self,
        sa_pred: torch.Tensor,
        sr_probs: torch.Tensor,
    ) -> torch.Tensor:
        """swing_result × spray_angle の整合性ペナルティを計算する.

        Args:
            sa_pred: (N,) 正規化空間の spray_angle 予測
            sr_probs: (N, 3) swing_result のソフト確率
        """
        m = self.margin_sa

        # hit_into_play (cls 1): フェアゾーン内 [-45°, +45°] → はみ出しペナルティ
        penalty_hip = torch.relu((self.t_sa_min - m) - sa_pred) + torch.relu(sa_pred - (self.t_sa_max + m))

        # foul (cls 0): フェアゾーン外 → フェア内にいるとペナルティ
        # relu(sa - (min + m)) * relu((max - m) - sa) はフェア内部で > 0
        penalty_foul = torch.relu(sa_pred - (self.t_sa_min + m)) * torch.relu((self.t_sa_max - m) - sa_pred)

        loss = (
            sr_probs[:, _CLS_HIT_INTO_PLAY] * penalty_hip**2
            + sr_probs[:, _CLS_FOUL] * penalty_foul
        )
        return loss.mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """物理的整合性ペナルティを計算する.

        Args:
            outputs: モデル出力辞書（bb_type, swing_result, regression を含む）
            batch: データバッチ（bb_type, swing_result, reg_mask を含む）

        Returns:
            スカラーペナルティ損失
        """
        device = outputs["bb_type"].device
        reg_pred = self._get_reg_pred(outputs["regression"])  # (B, D)
        total = torch.tensor(0.0, device=device)

        # --- bb_type × launch_angle ---
        bt_valid = batch["bb_type"] >= 0
        la_reg_valid = batch["reg_mask"][:, self.la_idx] > 0.5
        la_mask = bt_valid & la_reg_valid

        if la_mask.any():
            la_pred = reg_pred[la_mask, self.la_idx]
            bb_probs = F.softmax(outputs["bb_type"][la_mask], dim=-1)
            total = total + self._launch_angle_penalty(la_pred, bb_probs)

        # --- swing_result × spray_angle ---
        if "swing_result" in outputs:
            sr_valid = batch["swing_result"] >= 0
            sa_reg_valid = batch["reg_mask"][:, self.sa_idx] > 0.5
            sa_mask = sr_valid & sa_reg_valid

            if sa_mask.any():
                sa_pred = reg_pred[sa_mask, self.sa_idx]
                sr_probs = F.softmax(outputs["swing_result"][sa_mask], dim=-1)
                total = total + self._spray_angle_penalty(sa_pred, sr_probs)

        return total
