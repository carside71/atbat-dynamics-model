"""マルチタスク損失の計算."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, TrainConfig


def _mdn_loss(
    mdn_out: dict[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MDN の負の対数尤度損失（マスク付き）.

    Args:
        mdn_out: {"pi": (B, K), "mu": (B, K, D), "sigma": (B, K, D)}
        targets: (B, D) 真値
        mask: (B, D) 有効フラグ
    """
    sample_mask = mask.all(dim=-1)  # (B,)
    if not sample_mask.any():
        return torch.tensor(0.0, device=targets.device)

    pi = mdn_out["pi"][sample_mask]  # (N, K)
    mu = mdn_out["mu"][sample_mask]  # (N, K, D)
    sigma = mdn_out["sigma"][sample_mask]  # (N, K, D)
    t = targets[sample_mask].unsqueeze(1)  # (N, 1, D)

    # 各成分のガウス対数尤度: sum over D dimensions
    log_prob = -0.5 * (((t - mu) / sigma) ** 2 + 2 * torch.log(sigma) + 1.8378770664093453)  # ln(2π)
    log_prob = log_prob.sum(dim=-1)  # (N, K)

    # log-sum-exp over components
    log_pi = torch.log(pi + 1e-8)
    log_likelihood = torch.logsumexp(log_pi + log_prob, dim=-1)  # (N,)

    return -log_likelihood.mean()


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    train_cfg: TrainConfig,
    loss_fn_sr: nn.Module | None = None,
    loss_fn_bt: nn.Module | None = None,
    physics_loss_fn: nn.Module | None = None,
    model_cfg: ModelConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """階層的マスク付き損失を計算する.

    outputs に含まれるキーのタスクのみ損失を計算する。
    model_scope="all" の場合は全キーが含まれ、既存と同一の動作となる。

    Args:
        outputs: モデルの出力辞書
        batch: データバッチ
        train_cfg: 学習設定
        loss_fn_sr: swing_result 用の損失関数（None の場合は標準 cross_entropy）
        loss_fn_bt: bb_type 用の損失関数（None の場合は標準 cross_entropy）
    """
    losses = {}
    first_val = next(iter(outputs.values()))
    if isinstance(first_val, dict):
        device = next(iter(first_val.values())).device
    else:
        device = first_val.device
    total = torch.tensor(0.0, device=device)

    # 1. swing_attempt (binary cross-entropy)
    if "swing_attempt" in outputs:
        loss_sa = F.binary_cross_entropy_with_logits(outputs["swing_attempt"], batch["swing_attempt"])
        losses["swing_attempt"] = loss_sa.item()
        total = total + train_cfg.loss_weight_swing_attempt * loss_sa

    # 2-3. swing_result / bb_type（マスク付き分類損失）
    def _masked_cls_loss(key: str, loss_fn: nn.Module | None, weight: float) -> None:
        nonlocal total
        if key not in outputs:
            return
        mask = batch[key] >= 0
        if mask.any():
            logits, targets = outputs[key][mask], batch[key][mask]
            loss = loss_fn(logits, targets) if loss_fn else F.cross_entropy(logits, targets, label_smoothing=train_cfg.label_smoothing)
        else:
            loss = torch.tensor(0.0, device=device)
        losses[key] = loss.item()
        total = total + weight * loss

    _masked_cls_loss("swing_result", loss_fn_sr, train_cfg.loss_weight_swing_result)
    _masked_cls_loss("bb_type", loss_fn_bt, train_cfg.loss_weight_bb_type)

    # 4. regression
    if "regression" in outputs:
        reg_mask = batch["reg_mask"]  # (B, D)
        reg_out = outputs["regression"]
        if isinstance(reg_out, dict) and "heatmap_2d" in reg_out:
            # Heatmap head
            from losses.heatmap import compute_heatmap_loss

            assert model_cfg is not None, "model_cfg is required for heatmap head loss"
            loss_reg, hm_details = compute_heatmap_loss(reg_out, batch, model_cfg, train_cfg)
            losses.update(hm_details)
        elif isinstance(reg_out, dict):
            loss_reg = _mdn_loss(reg_out, batch["reg_targets"], reg_mask)
        elif reg_mask.any():
            diff = (reg_out - batch["reg_targets"]) * reg_mask
            loss_reg = (diff**2).sum() / reg_mask.sum().clamp(min=1)
        else:
            loss_reg = torch.tensor(0.0, device=device)
        losses["regression"] = loss_reg.item()
        total = total + train_cfg.loss_weight_regression * loss_reg

    # 5. physics consistency loss
    if (
        physics_loss_fn is not None
        and train_cfg.loss_weight_physics > 0
        and "bb_type" in outputs
        and "regression" in outputs
    ):
        loss_phys = physics_loss_fn(outputs, batch)
        losses["physics"] = loss_phys.item()
        total = total + train_cfg.loss_weight_physics * loss_phys

    losses["total"] = total.item()

    return total, losses
