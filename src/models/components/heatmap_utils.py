"""ヒートマップの後処理ユーティリティ（NMS・デコード）."""

import torch
import torch.nn.functional as F


def nms_2d(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """2D ヒートマップから NMS でピークを検出する.

    max_pool2d で局所最大値を見つけ、元の値と一致するピクセルのみ残す。

    Args:
        heatmap: (B, 1, H, W)
        kernel_size: max pooling のカーネルサイズ

    Returns:
        (B, 1, H, W) ピーク以外が 0 のヒートマップ
    """
    pad = kernel_size // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def nms_1d(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """1D ヒートマップから NMS でピークを検出する.

    Args:
        heatmap: (B, 1, L)
        kernel_size: max pooling のカーネルサイズ

    Returns:
        (B, 1, L) ピーク以外が 0 のヒートマップ
    """
    pad = kernel_size // 2
    hmax = F.max_pool1d(heatmap, kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_heatmap_2d(
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    value_range: tuple[float, float],
    grid_h: int,
    grid_w: int,
    kernel_size: int = 3,
) -> torch.Tensor:
    """2D ヒートマップ + オフセットから (launch_angle, spray_angle) を復元する.

    1. NMS でピーク検出
    2. flatten して argmax でピーク座標 (row, col) 取得
    3. ピクセル中心の正規化値を計算
    4. オフセットを加算

    Args:
        heatmap: (B, 1, H, W)
        offset: (B, 2, H, W)  — channel 0: dy (launch_angle), channel 1: dx (spray_angle)
        value_range: 正規化値域 (min, max)
        grid_h: グリッド高さ
        grid_w: グリッド幅
        kernel_size: NMS カーネルサイズ

    Returns:
        (B, 2) — [launch_angle_norm, spray_angle_norm]
    """
    B = heatmap.size(0)
    nms_hm = nms_2d(heatmap, kernel_size)  # (B, 1, H, W)

    # flatten して argmax
    flat = nms_hm.view(B, -1)  # (B, H*W)
    max_idx = flat.argmax(dim=-1)  # (B,)

    row = max_idx // grid_w  # (B,)
    col = max_idx % grid_w   # (B,)

    # ピクセル中心の正規化値
    vmin, vmax = value_range
    bin_h = (vmax - vmin) / grid_h
    bin_w = (vmax - vmin) / grid_w
    center_y = vmin + (row.float() + 0.5) * bin_h  # launch_angle
    center_x = vmin + (col.float() + 0.5) * bin_w  # spray_angle

    # ピーク位置でのオフセットを取得
    batch_idx = torch.arange(B, device=heatmap.device)
    off_y = offset[batch_idx, 0, row, col]  # (B,)
    off_x = offset[batch_idx, 1, row, col]  # (B,)

    pred_y = center_y + off_y
    pred_x = center_x + off_x

    return torch.stack([pred_y, pred_x], dim=-1)  # (B, 2)


def decode_heatmap_1d(
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    value_range: tuple[float, float],
    num_bins: int,
    kernel_size: int = 3,
) -> torch.Tensor:
    """1D ヒートマップ + オフセットから値を復元する.

    Args:
        heatmap: (B, 1, L)
        offset: (B, 1, L)
        value_range: 正規化値域 (min, max)
        num_bins: ビン数
        kernel_size: NMS カーネルサイズ

    Returns:
        (B,)
    """
    B = heatmap.size(0)
    nms_hm = nms_1d(heatmap, kernel_size)  # (B, 1, L)

    flat = nms_hm.view(B, -1)  # (B, L)
    max_idx = flat.argmax(dim=-1)  # (B,)

    vmin, vmax = value_range
    bin_size = (vmax - vmin) / num_bins
    center = vmin + (max_idx.float() + 0.5) * bin_size  # (B,)

    batch_idx = torch.arange(B, device=heatmap.device)
    off = offset[batch_idx, 0, max_idx]  # (B,)

    return center + off
