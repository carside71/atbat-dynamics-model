"""PhysicsLoss のテスト."""

import pytest
import torch

from config import TrainConfig
from losses.multi_task import compute_loss
from losses.physics import PhysicsLoss


# テスト用の正規化パラメータ（launch_angle: mean=20, std=15, spray_angle: mean=0, std=30）
_REG_NORM_STATS = {
    "launch_speed": (90.0, 10.0),
    "launch_angle": (20.0, 15.0),
    "hit_distance_sc": (200.0, 80.0),
    "spray_angle": (0.0, 30.0),
}
_TARGET_REG_COLUMNS = ["launch_speed", "launch_angle", "hit_distance_sc", "spray_angle"]


@pytest.fixture()
def physics_loss_fn():
    return PhysicsLoss(_REG_NORM_STATS, _TARGET_REG_COLUMNS, margin=2.0)


def _make_batch(B, bb_type_vals, sr_vals=None, reg_la_raw=None, reg_sa_raw=None):
    """テスト用のバッチを生成する.

    reg_la_raw / reg_sa_raw は生角度（度）で指定し、内部で正規化する。
    """
    la_mean, la_std = _REG_NORM_STATS["launch_angle"]
    sa_mean, sa_std = _REG_NORM_STATS["spray_angle"]

    reg_targets = torch.zeros(B, 4)
    reg_mask = torch.zeros(B, 4)

    if reg_la_raw is not None:
        for i, v in enumerate(reg_la_raw):
            if v is not None:
                reg_targets[i, 1] = (v - la_mean) / la_std
                reg_mask[i, 1] = 1.0

    if reg_sa_raw is not None:
        for i, v in enumerate(reg_sa_raw):
            if v is not None:
                reg_targets[i, 3] = (v - sa_mean) / sa_std
                reg_mask[i, 3] = 1.0

    if sr_vals is None:
        sr_vals = [-1] * B

    return {
        "bb_type": torch.tensor(bb_type_vals, dtype=torch.long),
        "swing_result": torch.tensor(sr_vals, dtype=torch.long),
        "reg_targets": reg_targets,
        "reg_mask": reg_mask,
    }


def _make_outputs(B, bb_logits, reg_la_raw, sr_logits=None, reg_sa_raw=None):
    """テスト用の出力辞書を生成する（回帰値は生角度で指定、正規化して返す）."""
    la_mean, la_std = _REG_NORM_STATS["launch_angle"]
    sa_mean, sa_std = _REG_NORM_STATS["spray_angle"]

    reg = torch.zeros(B, 4)
    for i, v in enumerate(reg_la_raw):
        reg[i, 1] = (v - la_mean) / la_std

    if reg_sa_raw is not None:
        for i, v in enumerate(reg_sa_raw):
            reg[i, 3] = (v - sa_mean) / sa_std

    out = {
        "bb_type": torch.tensor(bb_logits, dtype=torch.float32),
        "regression": reg,
    }
    if sr_logits is not None:
        out["swing_result"] = torch.tensor(sr_logits, dtype=torch.float32)
    return out


class TestPhysicsLoss:
    def test_normalized_thresholds(self, physics_loss_fn):
        """正規化閾値が正しく計算されていることを確認."""
        la_mean, la_std = 20.0, 15.0
        assert abs(physics_loss_fn.t_gb_ld.item() - (10.0 - la_mean) / la_std) < 1e-6
        assert abs(physics_loss_fn.t_ld_fb.item() - (25.0 - la_mean) / la_std) < 1e-6
        assert abs(physics_loss_fn.t_fb_pu.item() - (50.0 - la_mean) / la_std) < 1e-6

        sa_mean, sa_std = 0.0, 30.0
        assert abs(physics_loss_fn.t_sa_min.item() - (-45.0 - sa_mean) / sa_std) < 1e-6
        assert abs(physics_loss_fn.t_sa_max.item() - (45.0 - sa_mean) / sa_std) < 1e-6

    def test_consistent_gb_no_penalty(self, physics_loss_fn):
        """ground_ball + LA=5° (範囲内) → loss ≈ 0."""
        B = 2
        # 強い ground_ball logits + LA=5° (< 10°)
        batch = _make_batch(B, bb_type_vals=[0, 0], reg_la_raw=[5.0, 5.0])
        outputs = _make_outputs(B, bb_logits=[[10.0, -10, -10, -10]] * B, reg_la_raw=[5.0, 5.0])

        loss = physics_loss_fn(outputs, batch)
        assert loss.item() < 1e-4

    def test_inconsistent_gb_has_penalty(self, physics_loss_fn):
        """ground_ball + LA=40° (範囲外) → loss > 0."""
        B = 2
        batch = _make_batch(B, bb_type_vals=[0, 0], reg_la_raw=[40.0, 40.0])
        outputs = _make_outputs(B, bb_logits=[[10.0, -10, -10, -10]] * B, reg_la_raw=[40.0, 40.0])

        loss = physics_loss_fn(outputs, batch)
        assert loss.item() > 0.1

    def test_consistent_popup_no_penalty(self, physics_loss_fn):
        """popup + LA=60° (範囲内) → loss ≈ 0."""
        B = 1
        batch = _make_batch(B, bb_type_vals=[3], reg_la_raw=[60.0])
        outputs = _make_outputs(B, bb_logits=[[- 10, -10, -10, 10.0]], reg_la_raw=[60.0])

        loss = physics_loss_fn(outputs, batch)
        assert loss.item() < 1e-4

    def test_inconsistent_popup_has_penalty(self, physics_loss_fn):
        """popup + LA=10° (範囲外) → loss > 0."""
        B = 1
        batch = _make_batch(B, bb_type_vals=[3], reg_la_raw=[10.0])
        outputs = _make_outputs(B, bb_logits=[[-10, -10, -10, 10.0]], reg_la_raw=[10.0])

        loss = physics_loss_fn(outputs, batch)
        assert loss.item() > 0.1

    def test_all_masked_returns_zero(self, physics_loss_fn):
        """bb_type 全て -1（無効） → loss = 0."""
        B = 4
        batch = _make_batch(B, bb_type_vals=[-1, -1, -1, -1], reg_la_raw=[40.0] * B)
        outputs = _make_outputs(B, bb_logits=[[10.0, -10, -10, -10]] * B, reg_la_raw=[40.0] * B)

        loss = physics_loss_fn(outputs, batch)
        assert loss.item() == 0.0

    def test_spray_angle_hip_consistent(self, physics_loss_fn):
        """hit_into_play + SA=10° (フェア内) → spray loss ≈ 0."""
        B = 1
        batch = _make_batch(B, bb_type_vals=[1], sr_vals=[1], reg_la_raw=[30.0], reg_sa_raw=[10.0])
        outputs = _make_outputs(
            B, bb_logits=[[-10, 10.0, -10, -10]], reg_la_raw=[30.0],
            sr_logits=[[-10, 10.0, -10]], reg_sa_raw=[10.0],
        )

        loss = physics_loss_fn(outputs, batch)
        # bb_type も整合的なので全体で loss ≈ 0
        assert loss.item() < 1e-3

    def test_spray_angle_foul_in_fair_zone_penalty(self, physics_loss_fn):
        """foul + SA=0° (フェア内) → penalty > 0."""
        B = 1
        batch = _make_batch(B, bb_type_vals=[-1], sr_vals=[0], reg_la_raw=[None], reg_sa_raw=[0.0])
        outputs = _make_outputs(
            B, bb_logits=[[-10, -10, -10, -10]], reg_la_raw=[20.0],
            sr_logits=[[10.0, -10, -10]], reg_sa_raw=[0.0],
        )

        loss = physics_loss_fn(outputs, batch)
        assert loss.item() > 0.01

    def test_mdn_head(self, physics_loss_fn):
        """MDN 形式の regression 出力に対応できることを確認."""
        B = 2
        K = 3
        D = 4
        la_mean, la_std = _REG_NORM_STATS["launch_angle"]

        # MDN 出力: 全コンポーネントが LA=40° を指す → ground_ball と矛盾
        mu = torch.zeros(B, K, D)
        mu[:, :, 1] = (40.0 - la_mean) / la_std  # launch_angle

        mdn_out = {
            "pi": torch.ones(B, K) / K,  # 均等混合
            "mu": mu,
            "sigma": torch.ones(B, K, D) * 0.1,
        }

        batch = _make_batch(B, bb_type_vals=[0, 0], reg_la_raw=[40.0, 40.0])
        outputs = {
            "bb_type": torch.tensor([[10.0, -10, -10, -10]] * B),
            "regression": mdn_out,
        }

        loss = physics_loss_fn(outputs, batch)
        assert loss.item() > 0.1

    def test_gradient_flow(self, physics_loss_fn):
        """backward() で bb_type logits と regression の両方に grad が伝播."""
        B = 2
        bb_logits = torch.tensor([[10.0, -10, -10, -10]] * B, requires_grad=True)
        reg = torch.zeros(B, 4, requires_grad=True)

        la_mean, la_std = _REG_NORM_STATS["launch_angle"]
        with torch.no_grad():
            reg_data = reg.clone()
            reg_data[:, 1] = (40.0 - la_mean) / la_std  # LA=40° (ground_ball と矛盾)
        reg = reg_data.clone().requires_grad_(True)

        batch = _make_batch(B, bb_type_vals=[0, 0], reg_la_raw=[40.0, 40.0])
        outputs = {"bb_type": bb_logits, "regression": reg}

        loss = physics_loss_fn(outputs, batch)
        loss.backward()

        assert bb_logits.grad is not None
        assert reg.grad is not None
        assert bb_logits.grad.abs().sum() > 0
        assert reg.grad.abs().sum() > 0


class TestComputeLossWithPhysics:
    def test_physics_key_in_losses(self, fake_batch):
        """physics_loss_fn を渡すと losses に 'physics' キーが含まれる."""
        train_cfg = TrainConfig(loss_weight_physics=0.01)
        physics_fn = PhysicsLoss(_REG_NORM_STATS, _TARGET_REG_COLUMNS)

        B = fake_batch["swing_attempt"].shape[0]
        outputs = {
            "swing_attempt": torch.randn(B),
            "swing_result": torch.randn(B, 3),
            "bb_type": torch.randn(B, 4),
            "regression": torch.randn(B, 5),
        }

        total, losses = compute_loss(outputs, fake_batch, train_cfg, physics_loss_fn=physics_fn)
        assert "physics" in losses

    def test_no_physics_when_none(self, fake_batch):
        """physics_loss_fn=None → 'physics' キーなし（後方互換）."""
        train_cfg = TrainConfig()
        B = fake_batch["swing_attempt"].shape[0]
        outputs = {
            "swing_attempt": torch.randn(B),
            "swing_result": torch.randn(B, 3),
            "bb_type": torch.randn(B, 4),
            "regression": torch.randn(B, 5),
        }

        total, losses = compute_loss(outputs, fake_batch, train_cfg)
        assert "physics" not in losses

    def test_no_physics_when_weight_zero(self, fake_batch):
        """loss_weight_physics=0.0 → 'physics' キーなし."""
        train_cfg = TrainConfig(loss_weight_physics=0.0)
        physics_fn = PhysicsLoss(_REG_NORM_STATS, _TARGET_REG_COLUMNS)

        B = fake_batch["swing_attempt"].shape[0]
        outputs = {
            "swing_attempt": torch.randn(B),
            "swing_result": torch.randn(B, 3),
            "bb_type": torch.randn(B, 4),
            "regression": torch.randn(B, 5),
        }

        total, losses = compute_loss(outputs, fake_batch, train_cfg, physics_loss_fn=physics_fn)
        assert "physics" not in losses
