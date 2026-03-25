"""Heatmap Head の動作確認テスト."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_heatmap_head_forward_shape():
    """HeatmapHead の forward が正しい shape の dict を返すことを確認."""
    from config import ModelConfig
    from models.components.heatmap_head import HeatmapHead

    cfg = ModelConfig(
        regression_head_type="heatmap",
        heatmap_grid_h=64,
        heatmap_grid_w=64,
        heatmap_num_bins=64,
        heatmap_intermediate_dim=256,
        dropout=0.1,
    )
    head = HeatmapHead(in_dim=128, cfg=cfg)

    B = 4
    x = torch.randn(B, 128)
    out = head(x)

    assert out["heatmap_2d"].shape == (B, 1, 64, 64), f"Got {out['heatmap_2d'].shape}"
    assert out["offset_2d"].shape == (B, 2, 64, 64), f"Got {out['offset_2d'].shape}"
    assert out["heatmap_launch_speed"].shape == (B, 1, 64), f"Got {out['heatmap_launch_speed'].shape}"
    assert out["offset_launch_speed"].shape == (B, 1, 64), f"Got {out['offset_launch_speed'].shape}"
    assert out["heatmap_hit_distance"].shape == (B, 1, 64), f"Got {out['heatmap_hit_distance'].shape}"
    assert out["offset_hit_distance"].shape == (B, 1, 64), f"Got {out['offset_hit_distance'].shape}"

    # heatmap は sigmoid 適用済みなので [0, 1] の範囲
    assert out["heatmap_2d"].min() >= 0.0
    assert out["heatmap_2d"].max() <= 1.0
    print("PASS: test_heatmap_head_forward_shape")


def test_gt_heatmap_2d():
    """2D GT ヒートマップが正しいガウス形状になることを確認."""
    from losses.heatmap import generate_gt_heatmap_2d

    B = 2
    targets = torch.tensor([[0.0, 0.0], [1.0, -1.0]])  # [launch_angle, spray_angle]
    mask = torch.ones(B, 2)

    gt_hm, gt_off, gt_idx, sm = generate_gt_heatmap_2d(
        targets, mask, grid_h=64, grid_w=64, value_range=(-4.0, 4.0), sigma=2.0,
    )

    assert gt_hm.shape == (B, 1, 64, 64), f"Got {gt_hm.shape}"
    assert gt_off.shape == (B, 2, 64, 64), f"Got {gt_off.shape}"
    assert gt_idx.shape == (B, 2), f"Got {gt_idx.shape}"
    assert sm.sum() == B  # 全サンプル有効

    # ガウスのピークが GT ピクセルにあること
    for b in range(B):
        row, col = gt_idx[b, 0].item(), gt_idx[b, 1].item()
        assert gt_hm[b, 0, row, col] == 1.0, f"Peak should be 1.0, got {gt_hm[b, 0, row, col]}"

    # ガウスの値は [0, 1] の範囲
    assert gt_hm.min() >= 0.0
    assert gt_hm.max() <= 1.0
    print("PASS: test_gt_heatmap_2d")


def test_gt_heatmap_1d():
    """1D GT ヒートマップが正しいガウス形状になることを確認."""
    from losses.heatmap import generate_gt_heatmap_1d

    B = 3
    targets = torch.tensor([0.0, 2.0, -2.0])
    mask = torch.ones(B)

    gt_hm, gt_off, gt_idx, sm = generate_gt_heatmap_1d(
        targets, mask, num_bins=64, value_range=(-4.0, 4.0), sigma=2.0,
    )

    assert gt_hm.shape == (B, 1, 64), f"Got {gt_hm.shape}"
    assert gt_off.shape == (B, 1, 64), f"Got {gt_off.shape}"
    assert gt_idx.shape == (B,), f"Got {gt_idx.shape}"
    assert sm.sum() == B

    # ピーク位置のチェック
    for b in range(B):
        idx = gt_idx[b].item()
        assert gt_hm[b, 0, idx] == 1.0, f"Peak should be 1.0, got {gt_hm[b, 0, idx]}"

    print("PASS: test_gt_heatmap_1d")


def test_nms_2d():
    """2D NMS が正しくピークを検出することを確認."""
    from models.components.heatmap_utils import nms_2d

    hm = torch.zeros(1, 1, 8, 8)
    # ピークを 2 つ設定
    hm[0, 0, 2, 3] = 0.9
    hm[0, 0, 6, 5] = 0.8
    # ピーク周辺に低い値
    hm[0, 0, 2, 4] = 0.5
    hm[0, 0, 6, 4] = 0.3

    nms_hm = nms_2d(hm, kernel_size=3)

    # ピーク位置は残る
    assert nms_hm[0, 0, 2, 3] == 0.9, f"Peak at (2,3) should remain, got {nms_hm[0, 0, 2, 3]}"
    assert nms_hm[0, 0, 6, 5] == 0.8, f"Peak at (6,5) should remain, got {nms_hm[0, 0, 6, 5]}"
    # 非ピーク位置は抑制される
    assert nms_hm[0, 0, 2, 4] == 0.0, f"Non-peak at (2,4) should be 0, got {nms_hm[0, 0, 2, 4]}"
    assert nms_hm[0, 0, 6, 4] == 0.0, f"Non-peak at (6,4) should be 0, got {nms_hm[0, 0, 6, 4]}"

    print("PASS: test_nms_2d")


def test_nms_1d():
    """1D NMS が正しくピークを検出することを確認."""
    from models.components.heatmap_utils import nms_1d

    hm = torch.zeros(1, 1, 10)
    hm[0, 0, 3] = 0.9
    hm[0, 0, 7] = 0.7
    hm[0, 0, 4] = 0.5
    hm[0, 0, 8] = 0.3

    nms_hm = nms_1d(hm, kernel_size=3)

    assert nms_hm[0, 0, 3] == 0.9
    assert nms_hm[0, 0, 7] == 0.7
    assert nms_hm[0, 0, 4] == 0.0
    assert nms_hm[0, 0, 8] == 0.0

    print("PASS: test_nms_1d")


def test_encode_decode_roundtrip_2d():
    """2D: GT 値 → ヒートマップ生成 → decode で元の値に近い値が復元されることを確認."""
    from losses.heatmap import generate_gt_heatmap_2d
    from models.components.heatmap_utils import decode_heatmap_2d

    B = 4
    value_range = (-4.0, 4.0)
    grid_h, grid_w = 64, 64

    # ランダムな GT 値
    torch.manual_seed(42)
    targets = torch.randn(B, 2)  # [launch_angle, spray_angle]
    mask = torch.ones(B, 2)

    gt_hm, gt_off, gt_idx, sm = generate_gt_heatmap_2d(
        targets, mask, grid_h, grid_w, value_range, sigma=2.0,
    )

    # GT ヒートマップとオフセットを使ってデコード
    decoded = decode_heatmap_2d(gt_hm, gt_off, value_range, grid_h, grid_w)

    # 元の値との誤差が 1 ビン幅以内
    bin_size = (value_range[1] - value_range[0]) / grid_h
    error = (decoded - targets).abs()
    max_error = error.max().item()
    assert max_error < bin_size * 2, f"Max roundtrip error {max_error:.4f} > 2 * bin_size {bin_size * 2:.4f}"
    print(f"PASS: test_encode_decode_roundtrip_2d (max error: {max_error:.4f}, bin_size: {bin_size:.4f})")


def test_encode_decode_roundtrip_1d():
    """1D: GT 値 → ヒートマップ生成 → decode で元の値に近い値が復元されることを確認."""
    from losses.heatmap import generate_gt_heatmap_1d
    from models.components.heatmap_utils import decode_heatmap_1d

    B = 4
    value_range = (-4.0, 4.0)
    num_bins = 64

    torch.manual_seed(42)
    targets = torch.randn(B)
    mask = torch.ones(B)

    gt_hm, gt_off, gt_idx, sm = generate_gt_heatmap_1d(
        targets, mask, num_bins, value_range, sigma=2.0,
    )

    decoded = decode_heatmap_1d(gt_hm, gt_off, value_range, num_bins)

    bin_size = (value_range[1] - value_range[0]) / num_bins
    error = (decoded - targets).abs()
    max_error = error.max().item()
    assert max_error < bin_size * 2, f"Max roundtrip error {max_error:.4f} > 2 * bin_size {bin_size * 2:.4f}"
    print(f"PASS: test_encode_decode_roundtrip_1d (max error: {max_error:.4f}, bin_size: {bin_size:.4f})")


def test_heatmap_loss_computation():
    """ヒートマップ損失の計算が正常に動作することを確認."""
    from config import ModelConfig, TrainConfig
    from losses.heatmap import compute_heatmap_loss
    from models.components.heatmap_head import HeatmapHead

    cfg = ModelConfig(
        regression_head_type="heatmap",
        heatmap_grid_h=32,
        heatmap_grid_w=32,
        heatmap_num_bins=32,
        heatmap_intermediate_dim=64,
        dropout=0.1,
    )
    train_cfg = TrainConfig()

    head = HeatmapHead(in_dim=64, cfg=cfg)

    B = 8
    x = torch.randn(B, 64)
    outputs = head(x)

    batch = {
        "reg_targets": torch.randn(B, 4),  # [launch_speed, launch_angle, hit_distance_sc, spray_angle]
        "reg_mask": torch.ones(B, 4),
    }

    loss, details = compute_heatmap_loss(outputs, batch, cfg, train_cfg)

    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    assert loss.requires_grad, "Loss should require grad"

    expected_keys = {"hm_2d", "off_2d", "hm_ls", "off_ls", "hm_hd", "off_hd"}
    assert expected_keys.issubset(details.keys()), f"Missing keys: {expected_keys - details.keys()}"

    print(f"PASS: test_heatmap_loss_computation (loss={loss.item():.4f})")
    for k, v in details.items():
        print(f"  {k}: {v:.4f}")


def test_head_strategy_integration():
    """head_strategies で heatmap ヘッドが正しく構築されることを確認."""
    from config import ModelConfig
    from models.components.head_strategies import _build_regression_head

    cfg = ModelConfig(
        regression_head_type="heatmap",
        heatmap_grid_h=32,
        heatmap_grid_w=32,
        heatmap_num_bins=32,
        heatmap_intermediate_dim=64,
        dropout=0.1,
    )
    head = _build_regression_head(cfg, in_dim=64)

    B = 4
    x = torch.randn(B, 64)
    out = head(x)

    assert isinstance(out, dict)
    assert "heatmap_2d" in out
    print("PASS: test_head_strategy_integration")


def test_backward_pass():
    """勾配が正常に伝播することを確認."""
    from config import ModelConfig, TrainConfig
    from losses.heatmap import compute_heatmap_loss
    from models.components.heatmap_head import HeatmapHead

    cfg = ModelConfig(
        regression_head_type="heatmap",
        heatmap_grid_h=32,
        heatmap_grid_w=32,
        heatmap_num_bins=32,
        heatmap_intermediate_dim=64,
        dropout=0.1,
    )
    train_cfg = TrainConfig()

    head = HeatmapHead(in_dim=64, cfg=cfg)

    B = 4
    x = torch.randn(B, 64, requires_grad=True)
    outputs = head(x)

    batch = {
        "reg_targets": torch.randn(B, 4),
        "reg_mask": torch.ones(B, 4),
    }

    loss, _ = compute_heatmap_loss(outputs, batch, cfg, train_cfg)
    loss.backward()

    # 全パラメータに勾配があること
    for name, param in head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"

    print("PASS: test_backward_pass")


if __name__ == "__main__":
    test_nms_2d()
    test_nms_1d()
    test_gt_heatmap_2d()
    test_gt_heatmap_1d()
    test_encode_decode_roundtrip_2d()
    test_encode_decode_roundtrip_1d()
    test_heatmap_head_forward_shape()
    test_heatmap_loss_computation()
    test_head_strategy_integration()
    test_backward_pass()
    print("\nAll tests passed!")
