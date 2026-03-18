"""学習・評価スクリプト."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# src ディレクトリを起点にインポート
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DataConfig, ModelConfig, TrainConfig, load_config
from dataset import (
    StatcastDataset,
    compute_normalization_stats,
    load_all_parquet_files,
    load_split_at_bat_ids,
    load_stats,
)
from utils.logging import tee_logging
from utils.model_io import build_model, save_model_config


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
    # 全次元が有効なサンプルのみを対象にする
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """階層的マスク付き損失を計算する."""
    losses = {}

    # 1. swing_attempt (binary cross-entropy)
    loss_sa = nn.functional.binary_cross_entropy_with_logits(outputs["swing_attempt"], batch["swing_attempt"])
    losses["swing_attempt"] = loss_sa.item()

    # 2. swing_result (cross-entropy, swing_attempt=True のサンプルのみ)
    sr_mask = batch["swing_result"] >= 0
    if sr_mask.any():
        loss_sr = nn.functional.cross_entropy(outputs["swing_result"][sr_mask], batch["swing_result"][sr_mask])
    else:
        loss_sr = torch.tensor(0.0, device=outputs["swing_attempt"].device)
    losses["swing_result"] = loss_sr.item()

    # 3. bb_type (cross-entropy, swing_result==1 のサンプルのみ)
    bt_mask = batch["bb_type"] >= 0
    if bt_mask.any():
        loss_bt = nn.functional.cross_entropy(outputs["bb_type"][bt_mask], batch["bb_type"][bt_mask])
    else:
        loss_bt = torch.tensor(0.0, device=outputs["swing_attempt"].device)
    losses["bb_type"] = loss_bt.item()

    # 4. regression
    reg_mask = batch["reg_mask"]  # (B, 3)
    reg_out = outputs["regression"]
    if isinstance(reg_out, dict):
        # MDN: 負の対数尤度
        loss_reg = _mdn_loss(reg_out, batch["reg_targets"], reg_mask)
    elif reg_mask.any():
        # 通常の MSE
        diff = (reg_out - batch["reg_targets"]) * reg_mask
        loss_reg = (diff**2).sum() / reg_mask.sum().clamp(min=1)
    else:
        loss_reg = torch.tensor(0.0, device=outputs["swing_attempt"].device)
    losses["regression"] = loss_reg.item()

    total = (
        train_cfg.loss_weight_swing_attempt * loss_sa
        + train_cfg.loss_weight_swing_result * loss_sr
        + train_cfg.loss_weight_bb_type * loss_bt
        + train_cfg.loss_weight_regression * loss_reg
    )
    losses["total"] = total.item()

    return total, losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    device: torch.device,
) -> dict[str, float]:
    """検証データで評価を行い、損失とメトリクスを返す."""
    model.eval()
    total_losses = {}
    n_batches = 0

    # accuracy 用の集計
    sa_correct, sa_total = 0, 0
    sr_correct, sr_total = 0, 0
    bt_correct, bt_total = 0, 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        cat_dict = {col: batch[col] for col in data_cfg.categorical_features}
        outputs = model(cat_dict, batch["cont"], batch["ord"])

        _, losses = compute_loss(outputs, batch, train_cfg)
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n_batches += 1

        # swing_attempt accuracy
        pred_sa = (outputs["swing_attempt"].sigmoid() > 0.5).float()
        sa_correct += (pred_sa == batch["swing_attempt"]).sum().item()
        sa_total += len(pred_sa)

        # swing_result accuracy
        sr_mask = batch["swing_result"] >= 0
        if sr_mask.any():
            pred_sr = outputs["swing_result"][sr_mask].argmax(dim=-1)
            sr_correct += (pred_sr == batch["swing_result"][sr_mask]).sum().item()
            sr_total += sr_mask.sum().item()

        # bb_type accuracy
        bt_mask = batch["bb_type"] >= 0
        if bt_mask.any():
            pred_bt = outputs["bb_type"][bt_mask].argmax(dim=-1)
            bt_correct += (pred_bt == batch["bb_type"][bt_mask]).sum().item()
            bt_total += bt_mask.sum().item()

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    avg_losses["acc_swing_attempt"] = sa_correct / max(sa_total, 1)
    avg_losses["acc_swing_result"] = sr_correct / max(sr_total, 1)
    avg_losses["acc_bb_type"] = bt_correct / max(bt_total, 1)

    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="AtBat Dynamics Model Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    if args.config:
        data_cfg, model_cfg, train_cfg = load_config(args.config)
        config_name = Path(args.config).stem
    else:
        data_cfg = DataConfig()
        model_cfg = ModelConfig()
        train_cfg = TrainConfig()
        config_name = "default"

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_dir = data_cfg.output_dir / config_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ログを端末とファイルの両方に出力
    with tee_logging(output_dir / "train.log"):
        _train(data_cfg, model_cfg, train_cfg, output_dir)


def _train(data_cfg, model_cfg, train_cfg, output_dir):
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # === データ読み込み ===
    print("Loading stats...")
    stats = load_stats(data_cfg.stats_dir)

    print("Loading data...")
    all_df = load_all_parquet_files(data_cfg.data_dir)
    print(f"  Total samples: {len(all_df):,}")

    print("Splitting by at_bat_id...")
    train_ids = load_split_at_bat_ids(data_cfg.split_dir, "train")
    val_ids = load_split_at_bat_ids(data_cfg.split_dir, "val")
    train_df = all_df[all_df["at_bat_id"].isin(train_ids)].drop(columns=["at_bat_id"]).reset_index(drop=True)
    val_df = all_df[all_df["at_bat_id"].isin(val_ids)].drop(columns=["at_bat_id"]).reset_index(drop=True)
    del all_df
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Val samples: {len(val_df):,}")

    # === 正規化パラメータを訓練データから計算 ===
    print("Computing normalization stats...")
    norm_stats = compute_normalization_stats(train_df, data_cfg.continuous_features)
    reg_norm_stats = compute_normalization_stats(train_df, data_cfg.target_reg)

    # 正規化パラメータを保存
    norm_params = {"input": norm_stats, "target": reg_norm_stats}
    with open(output_dir / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)

    # === Dataset & DataLoader ===
    print("Building datasets...")
    train_ds = StatcastDataset(train_df, data_cfg, norm_stats, reg_norm_stats)
    val_ds = StatcastDataset(val_df, data_cfg, norm_stats, reg_norm_stats)
    del train_df, val_df  # メモリ解放

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    # === モデル構築 ===
    print("Building model...")
    model = build_model(data_cfg, model_cfg, stats)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # モデル設定を保存
    save_model_config(model_cfg, data_cfg, output_dir)

    # === 学習 ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)

    best_val_loss = float("inf")
    history = []

    print("Starting training...")
    for epoch in range(1, train_cfg.num_epochs + 1):
        model.train()
        epoch_losses = {}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            cat_dict = {col: batch[col] for col in data_cfg.categorical_features}

            optimizer.zero_grad()
            outputs = model(cat_dict, batch["cont"], batch["ord"])
            loss, losses = compute_loss(outputs, batch, train_cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_batches += 1

            pbar.set_postfix(loss=f"{losses['total']:.4f}")

        scheduler.step()

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # 検証
        val_metrics = evaluate(model, val_loader, train_cfg, data_cfg, device)

        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        record.update({f"train_{k}": v for k, v in avg_train.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {avg_train['total']:.4f} | "
            f"Val Loss: {val_metrics['total']:.4f} | "
            f"Val SA Acc: {val_metrics['acc_swing_attempt']:.4f} | "
            f"Val SR Acc: {val_metrics['acc_swing_result']:.4f} | "
            f"Val BT Acc: {val_metrics['acc_bb_type']:.4f}"
        )

        # ベストモデル保存
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    # 最終モデル保存
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    # 学習履歴保存
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
