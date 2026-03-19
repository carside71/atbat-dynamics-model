"""学習・評価スクリプト."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# src ディレクトリを起点にインポート
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DataConfig, ModelConfig, TrainConfig, load_config
from datasets import (
    StatcastBatterHistDataset,
    StatcastDataset,
    StatcastSequenceDataset,
    compute_normalization_stats,
    load_all_parquet_files,
    load_split_at_bat_ids,
    load_stats,
)
from losses import FocalLoss, compute_loss
from utils.logging import tee_logging
from utils.model_io import build_model, save_model_config


def _build_class_weights(stats: dict[str, pd.DataFrame], key: str, device: torch.device) -> torch.Tensor:
    """stats からクラス頻度の逆数に基づく重みを計算する."""
    counts = stats[key]["count"].to_numpy(dtype=np.float64)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _model_forward(
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
            hist_pitch_mask=batch["hist_pitch_mask"],
            hist_atbat_mask=batch["hist_atbat_mask"],
        )
    return model(cat_dict, batch["cont"], batch["ord"], **kwargs)


def _build_loss_functions(
    train_cfg: TrainConfig,
    stats: dict[str, pd.DataFrame],
    device: torch.device,
) -> tuple[nn.Module | None, nn.Module | None]:
    """swing_result / bb_type 用の損失関数を構築する."""
    if train_cfg.focal_gamma == 0.0 and not train_cfg.use_class_weight and train_cfg.label_smoothing == 0.0:
        return None, None  # 標準 cross-entropy を使用

    sr_weight = _build_class_weights(stats, "swing_result", device) if train_cfg.use_class_weight else None
    bt_weight = _build_class_weights(stats, "bb_type", device) if train_cfg.use_class_weight else None

    loss_fn_sr = FocalLoss(gamma=train_cfg.focal_gamma, weight=sr_weight, label_smoothing=train_cfg.label_smoothing)
    loss_fn_bt = FocalLoss(gamma=train_cfg.focal_gamma, weight=bt_weight, label_smoothing=train_cfg.label_smoothing)
    return loss_fn_sr, loss_fn_bt


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    device: torch.device,
    loss_fn_sr: nn.Module | None = None,
    loss_fn_bt: nn.Module | None = None,
    use_seq: bool = False,
    use_batter_hist: bool = False,
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
        outputs = _model_forward(model, batch, data_cfg, use_seq, use_batter_hist)

        _, losses = compute_loss(outputs, batch, train_cfg, loss_fn_sr, loss_fn_bt)
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

    use_seq = model_cfg.max_seq_len > 0
    use_batter_hist = model_cfg.batter_hist_max_atbats > 0
    need_at_bat_id = use_seq or use_batter_hist
    # game_pk, batter は batter_hist Dataset で必要
    extra_cols_to_keep = ["game_pk"] if use_batter_hist else []

    if need_at_bat_id:
        train_df = all_df[all_df["at_bat_id"].isin(train_ids)].reset_index(drop=True)
        val_df = all_df[all_df["at_bat_id"].isin(val_ids)].reset_index(drop=True)
    else:
        drop_cols = ["at_bat_id"] + [c for c in ["game_pk", "game_date"] if c in all_df.columns]
        train_df = (
            all_df[all_df["at_bat_id"].isin(train_ids)].drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        )
        val_df = (
            all_df[all_df["at_bat_id"].isin(val_ids)].drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        )
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
    if use_batter_hist:
        train_ds = StatcastBatterHistDataset(
            train_df,
            data_cfg,
            model_cfg.max_seq_len,
            model_cfg.batter_hist_max_atbats,
            model_cfg.batter_hist_max_pitches,
            norm_stats,
            reg_norm_stats,
        )
        val_ds = StatcastBatterHistDataset(
            val_df,
            data_cfg,
            model_cfg.max_seq_len,
            model_cfg.batter_hist_max_atbats,
            model_cfg.batter_hist_max_pitches,
            norm_stats,
            reg_norm_stats,
        )
    elif use_seq:
        train_ds = StatcastSequenceDataset(train_df, data_cfg, model_cfg.max_seq_len, norm_stats, reg_norm_stats)
        val_ds = StatcastSequenceDataset(val_df, data_cfg, model_cfg.max_seq_len, norm_stats, reg_norm_stats)
    else:
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

    # === 損失関数 ===
    loss_fn_sr, loss_fn_bt = _build_loss_functions(train_cfg, stats, device)
    if loss_fn_sr is not None:
        print(f"  Using FocalLoss (gamma={train_cfg.focal_gamma}, class_weight={train_cfg.use_class_weight})")
    if train_cfg.label_smoothing > 0:
        print(f"  Label Smoothing: {train_cfg.label_smoothing}")
    print(
        f"  Loss weights: SA={train_cfg.loss_weight_swing_attempt}, SR={train_cfg.loss_weight_swing_result}, "
        f"BT={train_cfg.loss_weight_bb_type}, Reg={train_cfg.loss_weight_regression}"
    )

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

            optimizer.zero_grad()
            outputs = _model_forward(model, batch, data_cfg, use_seq, use_batter_hist)
            loss, losses = compute_loss(outputs, batch, train_cfg, loss_fn_sr, loss_fn_bt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_batches += 1

            pbar.set_postfix(
                total=f"{losses['total']:.4f}",
                SA=f"{losses['swing_attempt']:.4f}",
                SR=f"{losses['swing_result']:.4f}",
                BT=f"{losses['bb_type']:.4f}",
                Reg=f"{losses['regression']:.4f}",
            )

        scheduler.step()

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # 検証
        val_metrics = evaluate(
            model, val_loader, train_cfg, data_cfg, device, loss_fn_sr, loss_fn_bt, use_seq, use_batter_hist
        )

        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        record.update({f"train_{k}": v for k, v in avg_train.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {avg_train['total']:.4f} "
            f"(SA={avg_train['swing_attempt']:.4f} SR={avg_train['swing_result']:.4f} "
            f"BT={avg_train['bb_type']:.4f} Reg={avg_train['regression']:.4f}) | "
            f"Val Loss: {val_metrics['total']:.4f} "
            f"(SA={val_metrics['swing_attempt']:.4f} SR={val_metrics['swing_result']:.4f} "
            f"BT={val_metrics['bb_type']:.4f} Reg={val_metrics['regression']:.4f}) | "
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
