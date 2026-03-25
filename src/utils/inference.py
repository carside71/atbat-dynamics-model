"""train / test 共有の推論ユーティリティ."""

import torch
import torch.nn as nn

from config import DataConfig


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """バッチ内の Tensor を指定デバイスに移動する."""
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def model_forward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    data_cfg: DataConfig,
    use_seq: bool,
    use_batter_hist: bool = False,
) -> dict[str, torch.Tensor]:
    """モデルの forward を呼び出す（シーケンス・打者履歴対応）."""
    cat_dict = {col: batch[col] for col in data_cfg.categorical_features}
    kwargs = {}
    if use_seq:
        kwargs.update(
            seq_pitch_type=batch["seq_pitch_type"],
            seq_cont=batch["seq_cont"],
            seq_swing_attempt=batch["seq_swing_attempt"],
            seq_swing_result=batch["seq_swing_result"],
            seq_mask=batch["seq_mask"],
        )
    if use_batter_hist:
        kwargs.update(
            hist_pitch_type=batch["hist_pitch_type"],
            hist_cont=batch["hist_cont"],
            hist_swing_attempt=batch["hist_swing_attempt"],
            hist_swing_result=batch["hist_swing_result"],
            hist_bb_type=batch["hist_bb_type"],
            hist_launch_speed=batch["hist_launch_speed"],
            hist_launch_angle=batch["hist_launch_angle"],
            hist_spray_angle=batch["hist_spray_angle"],
            hist_pitch_mask=batch["hist_pitch_mask"],
            hist_atbat_mask=batch["hist_atbat_mask"],
        )
    return model(cat_dict, batch["cont"], batch["ord"], **kwargs)
