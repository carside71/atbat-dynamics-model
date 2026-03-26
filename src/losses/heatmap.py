"""ヒートマップヘッド用の損失関数と GT ヒートマップ生成."""

import torch

from config import ModelConfig, TrainConfig


def generate_gt_heatmap_2d(
    targets: torch.Tensor,
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    value_range_h: tuple[float, float],
    value_range_w: tuple[float, float],
    sigma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GT 値から 2D ガウスヒートマップとオフセットマップを生成する.

    Args:
        targets: (B, 2) 正規化済み [launch_angle, spray_angle]
        mask: (B, 2) 有効フラグ
        grid_h: グリッド高さ
        grid_w: グリッド幅
        value_range_h: H 軸（launch_angle）の正規化値域 (min, max)
        value_range_w: W 軸（spray_angle）の正規化値域 (min, max)
        sigma: ガウス分布の標準偏差（ピクセル単位）

    Returns:
        gt_heatmap: (B, 1, H, W) ガウス分布ヒートマップ
        gt_offset: (B, 2, H, W) GT ピクセルでのオフセット (dy, dx)
        gt_indices: (B, 2) GT のピクセル座標 (row, col)
        sample_mask: (B,) 両次元とも有効なサンプルのマスク
    """
    B = targets.size(0)
    device = targets.device
    vmin_h, vmax_h = value_range_h
    vmin_w, vmax_w = value_range_w
    bin_h = (vmax_h - vmin_h) / grid_h
    bin_w = (vmax_w - vmin_w) / grid_w

    # 両次元が有効なサンプルのみ
    sample_mask = mask[:, 0] * mask[:, 1]  # (B,)

    # GT 値をピクセル座標に変換（連続値）
    # targets[:, 0] = launch_angle → row (y 軸)
    # targets[:, 1] = spray_angle → col (x 軸)
    ct_y = (targets[:, 0] - vmin_h) / bin_h  # (B,) 連続ピクセル座標
    ct_x = (targets[:, 1] - vmin_w) / bin_w  # (B,)

    # 整数ピクセル座標（中心ピクセル）
    ct_y_int = ct_y.long().clamp(0, grid_h - 1)
    ct_x_int = ct_x.long().clamp(0, grid_w - 1)
    gt_indices = torch.stack([ct_y_int, ct_x_int], dim=-1)  # (B, 2)

    # オフセット: GT 連続座標 − ピクセル中心
    off_y = ct_y - (ct_y_int.float() + 0.5)  # (B,)
    off_x = ct_x - (ct_x_int.float() + 0.5)  # (B,)

    # ピクセル単位に変換して bin_size を掛ける
    gt_offset = torch.zeros(B, 2, grid_h, grid_w, device=device)
    batch_idx = torch.arange(B, device=device)
    gt_offset[batch_idx, 0, ct_y_int, ct_x_int] = off_y * bin_h
    gt_offset[batch_idx, 1, ct_y_int, ct_x_int] = off_x * bin_w

    # ガウスヒートマップ生成
    y_grid = torch.arange(grid_h, device=device, dtype=torch.float32)
    x_grid = torch.arange(grid_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")  # (H, W)

    # (B, H, W) のガウス分布
    dy = yy.unsqueeze(0) - ct_y_int.float().view(B, 1, 1)  # (B, H, W)
    dx = xx.unsqueeze(0) - ct_x_int.float().view(B, 1, 1)  # (B, H, W)
    gaussian = torch.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))  # (B, H, W)

    # 無効サンプルはゼロに
    gaussian = gaussian * sample_mask.float().view(B, 1, 1)

    gt_heatmap = gaussian.unsqueeze(1)  # (B, 1, H, W)

    return gt_heatmap, gt_offset, gt_indices, sample_mask


def generate_gt_heatmap_1d(
    targets: torch.Tensor,
    mask: torch.Tensor,
    num_bins: int,
    value_range: tuple[float, float],
    sigma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GT 値から 1D ガウスヒートマップとオフセットを生成する.

    Args:
        targets: (B,) 正規化済み値
        mask: (B,) 有効フラグ
        num_bins: ビン数
        value_range: 正規化値域 (min, max)
        sigma: ガウス分布の標準偏差（ビン単位）

    Returns:
        gt_heatmap: (B, 1, L) ガウス分布
        gt_offset: (B, 1, L) GT ビンでのオフセット
        gt_indices: (B,) GT のビンインデックス
        sample_mask: (B,) 有効サンプルのマスク
    """
    B = targets.size(0)
    device = targets.device
    vmin, vmax = value_range
    bin_size = (vmax - vmin) / num_bins

    sample_mask = mask  # (B,)

    # 連続ビン座標
    ct = (targets - vmin) / bin_size  # (B,)
    ct_int = ct.long().clamp(0, num_bins - 1)  # (B,)

    # オフセット
    off = (ct - (ct_int.float() + 0.5)) * bin_size  # (B,)

    gt_offset = torch.zeros(B, 1, num_bins, device=device)
    batch_idx = torch.arange(B, device=device)
    gt_offset[batch_idx, 0, ct_int] = off

    # ガウスヒートマップ
    bins = torch.arange(num_bins, device=device, dtype=torch.float32)  # (L,)
    d = bins.unsqueeze(0) - ct_int.float().unsqueeze(1)  # (B, L)
    gaussian = torch.exp(-d ** 2 / (2 * sigma ** 2))  # (B, L)
    gaussian = gaussian * sample_mask.float().unsqueeze(1)

    gt_heatmap = gaussian.unsqueeze(1)  # (B, 1, L)

    return gt_heatmap, gt_offset, ct_int, sample_mask


def heatmap_focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sample_mask: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CenterNet スタイルの penalty-reduced focal loss.

    L = -1/N * sum(
        (1 - p)^alpha * log(p)                if y == 1
        (1 - y)^beta * p^alpha * log(1 - p)    if y < 1
    )

    Args:
        pred: 予測ヒートマップ (任意 shape、最後の次元は空間)
        gt: GT ヒートマップ (同じ shape)
        sample_mask: (B,) 有効サンプルのマスク
        alpha: focal loss の alpha
        beta: focal loss の beta

    Returns:
        スカラー損失
    """
    num_pos = sample_mask.sum().clamp(min=1)

    # 数値安定性のためクランプ
    pred = pred.clamp(min=1e-6, max=1 - 1e-6)

    # positive: GT == 1 のピクセル（実際にはガウスなので厳密には 1 にならないが、閾値で判定）
    pos_mask = gt.eq(1).float()
    neg_mask = gt.lt(1).float()

    # 無効サンプルをマスク
    # sample_mask を pred の shape に broadcast
    ndim = pred.ndim
    sm = sample_mask.float()
    for _ in range(ndim - 1):
        sm = sm.unsqueeze(-1)

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask * sm
    neg_loss = -((1 - gt) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask * sm

    loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


def heatmap_offset_loss(
    pred_offset: torch.Tensor,
    gt_offset: torch.Tensor,
    gt_indices: torch.Tensor,
    sample_mask: torch.Tensor,
    is_2d: bool = True,
) -> torch.Tensor:
    """GT ピクセル/ビン位置でのみ L1 loss を計算する.

    Args:
        pred_offset: (B, C, H, W) or (B, 1, L)
        gt_offset: 同じ shape
        gt_indices: (B, 2) for 2D or (B,) for 1D
        sample_mask: (B,) 有効サンプルのマスク
        is_2d: 2D かどうか

    Returns:
        スカラー損失
    """
    B = pred_offset.size(0)
    num_valid = sample_mask.sum().clamp(min=1)
    batch_idx = torch.arange(B, device=pred_offset.device)

    if is_2d:
        row = gt_indices[:, 0]  # (B,)
        col = gt_indices[:, 1]  # (B,)
        # GT ピクセルでの予測オフセットを取得
        pred_at_gt = pred_offset[batch_idx, :, row, col]  # (B, 2)
        gt_at_gt = gt_offset[batch_idx, :, row, col]      # (B, 2)
        diff = torch.abs(pred_at_gt - gt_at_gt) * sample_mask.float().unsqueeze(-1)
    else:
        bin_idx = gt_indices  # (B,)
        pred_at_gt = pred_offset[batch_idx, 0, bin_idx]  # (B,)
        gt_at_gt = gt_offset[batch_idx, 0, bin_idx]      # (B,)
        diff = torch.abs(pred_at_gt - gt_at_gt) * sample_mask.float()

    return diff.sum() / num_valid


def compute_heatmap_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """ヒートマップヘッド全体の損失を計算する.

    target_reg のデフォルト順: ["launch_speed", "launch_angle", "hit_distance_sc", "spray_angle"]
    - 2D ヘッド: launch_angle (idx=1), spray_angle (idx=3)
    - 1D launch_speed: idx=0
    - 1D hit_distance: idx=2

    Args:
        outputs: HeatmapHead の出力 dict
        batch: データバッチ（reg_targets, reg_mask を含む）
        model_cfg: モデル設定
        train_cfg: 学習設定

    Returns:
        total_loss: 合計損失
        loss_details: 損失内訳の dict
    """
    reg_targets = batch["reg_targets"]  # (B, D)
    reg_mask = batch["reg_mask"]        # (B, D)
    device = reg_targets.device

    range_la = tuple(model_cfg.heatmap_norm_range_launch_angle)
    range_sa = tuple(model_cfg.heatmap_norm_range_spray_angle)
    range_ls = tuple(model_cfg.heatmap_norm_range_launch_speed)
    range_hd = tuple(model_cfg.heatmap_norm_range_hit_distance)
    sigma = model_cfg.heatmap_sigma
    alpha = train_cfg.heatmap_focal_alpha
    beta = train_cfg.heatmap_focal_beta
    w_offset = train_cfg.heatmap_loss_weight_offset

    total_loss = torch.tensor(0.0, device=device)
    details: dict[str, float] = {}

    # --- 2D: launch_angle (idx=1) × spray_angle (idx=3) ---
    targets_2d = torch.stack([reg_targets[:, 1], reg_targets[:, 3]], dim=-1)  # (B, 2)
    mask_2d = torch.stack([reg_mask[:, 1], reg_mask[:, 3]], dim=-1)          # (B, 2)

    gt_hm_2d, gt_off_2d, gt_idx_2d, sm_2d = generate_gt_heatmap_2d(
        targets_2d, mask_2d,
        model_cfg.heatmap_grid_h, model_cfg.heatmap_grid_w,
        value_range_h=range_la, value_range_w=range_sa,
        sigma=sigma,
    )

    loss_hm_2d = heatmap_focal_loss(outputs["heatmap_2d"], gt_hm_2d, sm_2d, alpha, beta)
    loss_off_2d = heatmap_offset_loss(outputs["offset_2d"], gt_off_2d, gt_idx_2d, sm_2d, is_2d=True)

    details["hm_2d"] = loss_hm_2d.item()
    details["off_2d"] = loss_off_2d.item()
    total_loss = total_loss + loss_hm_2d + w_offset * loss_off_2d

    # --- 1D: launch_speed (idx=0) ---
    gt_hm_ls, gt_off_ls, gt_idx_ls, sm_ls = generate_gt_heatmap_1d(
        reg_targets[:, 0], reg_mask[:, 0],
        model_cfg.heatmap_num_bins, range_ls, sigma,
    )
    loss_hm_ls = heatmap_focal_loss(outputs["heatmap_launch_speed"], gt_hm_ls, sm_ls, alpha, beta)
    loss_off_ls = heatmap_offset_loss(outputs["offset_launch_speed"], gt_off_ls, gt_idx_ls, sm_ls, is_2d=False)

    details["hm_ls"] = loss_hm_ls.item()
    details["off_ls"] = loss_off_ls.item()
    total_loss = total_loss + loss_hm_ls + w_offset * loss_off_ls

    # --- 1D: hit_distance_sc (idx=2) ---
    gt_hm_hd, gt_off_hd, gt_idx_hd, sm_hd = generate_gt_heatmap_1d(
        reg_targets[:, 2], reg_mask[:, 2],
        model_cfg.heatmap_num_bins, range_hd, sigma,
    )
    loss_hm_hd = heatmap_focal_loss(outputs["heatmap_hit_distance"], gt_hm_hd, sm_hd, alpha, beta)
    loss_off_hd = heatmap_offset_loss(outputs["offset_hit_distance"], gt_off_hd, gt_idx_hd, sm_hd, is_2d=False)

    details["hm_hd"] = loss_hm_hd.item()
    details["off_hd"] = loss_off_hd.item()
    total_loss = total_loss + loss_hm_hd + w_offset * loss_off_hd

    return total_loss, details
