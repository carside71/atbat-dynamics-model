"""学習・評価スクリプト."""

import argparse
import json
import shutil
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

from config import DataConfig, ModelConfig, TrainConfig, load_config, validate_model_scope
from datasets import (
    compute_normalization_stats,
    create_dataset,
    load_all_parquet_files,
    load_split_at_bat_ids,
    load_stats,
)
from losses import FocalLoss, PhysicsLoss, compute_loss
from utils.inference import model_forward, move_batch_to_device
from utils.logging import tee_logging
from utils.model_io import build_model, save_model_config


def _build_class_weights(stats: dict[str, pd.DataFrame], key: str, device: torch.device) -> torch.Tensor:
    """stats からクラス頻度の逆数に基づく重みを計算する."""
    counts = stats[key]["count"].to_numpy(dtype=np.float64)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


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


def _format_loss_parts(metrics: dict[str, float], scope: str) -> str:
    """スコープに応じた損失文字列を構築する."""
    parts = []
    if scope in ("all", "swing_attempt", "classification"):
        parts.append(f"SA={metrics.get('swing_attempt', 0.0):.4f}")
    if scope in ("all", "outcome", "classification"):
        parts.append(f"SR={metrics.get('swing_result', 0.0):.4f}")
        parts.append(f"BT={metrics.get('bb_type', 0.0):.4f}")
    if scope in ("all", "outcome", "regression"):
        parts.append(f"Reg={metrics.get('regression', 0.0):.4f}")
    if "physics" in metrics:
        parts.append(f"Phys={metrics['physics']:.4f}")
    return " ".join(parts)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    device: torch.device,
    model_scope: str = "all",
    loss_fn_sr: nn.Module | None = None,
    loss_fn_bt: nn.Module | None = None,
    use_seq: bool = False,
    use_batter_hist: bool = False,
    physics_loss_fn: nn.Module | None = None,
    model_cfg: "ModelConfig | None" = None,
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
        batch = move_batch_to_device(batch, device)
        outputs = model_forward(model, batch, data_cfg, use_seq, use_batter_hist)

        _, losses = compute_loss(outputs, batch, train_cfg, loss_fn_sr, loss_fn_bt, physics_loss_fn, model_cfg=model_cfg)
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n_batches += 1

        # swing_attempt accuracy
        if "swing_attempt" in outputs:
            pred_sa = (outputs["swing_attempt"].sigmoid() > 0.5).float()
            sa_correct += (pred_sa == batch["swing_attempt"]).sum().item()
            sa_total += len(pred_sa)

        # swing_result accuracy
        if "swing_result" in outputs:
            sr_mask = batch["swing_result"] >= 0
            if sr_mask.any():
                pred_sr = outputs["swing_result"][sr_mask].argmax(dim=-1)
                sr_correct += (pred_sr == batch["swing_result"][sr_mask]).sum().item()
                sr_total += sr_mask.sum().item()

        # bb_type accuracy
        if "bb_type" in outputs:
            bt_mask = batch["bb_type"] >= 0
            if bt_mask.any():
                pred_bt = outputs["bb_type"][bt_mask].argmax(dim=-1)
                bt_correct += (pred_bt == batch["bb_type"][bt_mask]).sum().item()
                bt_total += bt_mask.sum().item()

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    if model_scope in ("all", "swing_attempt", "classification"):
        avg_losses["acc_swing_attempt"] = sa_correct / max(sa_total, 1)
    if model_scope in ("all", "outcome", "classification"):
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

    # 学習に使った YAML 設定ファイルを出力ディレクトリにコピー
    if args.config:
        shutil.copy2(args.config, output_dir / "config.yaml")

    # ログを端末とファイルの両方に出力
    with tee_logging(output_dir / "train.log"):
        _train(data_cfg, model_cfg, train_cfg, output_dir)


def _train(data_cfg, model_cfg, train_cfg, output_dir):
    validate_model_scope(model_cfg.model_scope)
    model_scope = model_cfg.model_scope

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model scope: {model_scope}")

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

    use_seq = model_cfg.pitch_seq_max_len > 0
    use_batter_hist = model_cfg.batter_hist_max_atbats > 0
    need_at_bat_id = use_seq or use_batter_hist

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

    # === outcome スコープ: swing_attempt=1 のサンプルのみに絞り込み ===
    if model_scope == "outcome":
        train_df = train_df[train_df["swing_attempt"] == 1].reset_index(drop=True)
        val_df = val_df[val_df["swing_attempt"] == 1].reset_index(drop=True)
        print("  Filtered to swing_attempt=1:")
        print(f"    Train samples: {len(train_df):,}")
        print(f"    Val samples: {len(val_df):,}")

    # === regression スコープ: YAML設定ベースのフィルタ ===
    if model_scope == "regression":
        if data_cfg.filter_swing_attempt:
            train_df = train_df[train_df["swing_attempt"] == 1].reset_index(drop=True)
            val_df = val_df[val_df["swing_attempt"] == 1].reset_index(drop=True)
            print("  Filtered to swing_attempt=1:")
            print(f"    Train samples: {len(train_df):,}")
            print(f"    Val samples: {len(val_df):,}")

        if data_cfg.reg_target_filter in ("any", "all"):
            reg_cols = data_cfg.target_reg
            if data_cfg.reg_target_filter == "any":
                cond = lambda df: df[reg_cols].notna().any(axis=1)
            else:
                cond = lambda df: df[reg_cols].notna().all(axis=1)
            train_df = train_df[cond(train_df)].reset_index(drop=True)
            val_df = val_df[cond(val_df)].reset_index(drop=True)
            print(f"  Filtered by reg_target_filter={data_cfg.reg_target_filter!r}:")
            print(f"    Train samples: {len(train_df):,}")
            print(f"    Val samples: {len(val_df):,}")

        for split_name, df in [("Train", train_df), ("Val", val_df)]:
            print(f"  {split_name} label distribution:")
            print(f"    swing_result: {df[data_cfg.target_cls_swing_result].value_counts().sort_index().to_dict()}")
            print(f"    bb_type: {df[data_cfg.target_cls_bb_type].value_counts().sort_index().to_dict()}")

    # === 正規化パラメータを訓練データから計算 ===
    print("Computing normalization stats...")
    norm_stats = compute_normalization_stats(train_df, data_cfg.continuous_features)
    reg_norm_stats = compute_normalization_stats(train_df, data_cfg.target_reg)

    # 正規化パラメータを保存
    norm_params = {"input": norm_stats, "target": reg_norm_stats}
    with open(output_dir / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)

    # === ヒートマップ物理範囲→正規化範囲の変換 ===
    if model_cfg.regression_head_type == "heatmap":
        # ターゲット名の順序を保存（設定モードの損失計算・デコードで使用）
        model_cfg.heatmap_target_reg = list(data_cfg.target_reg)

        _heatmap_range_map = [
            ("launch_speed", "heatmap_range_launch_speed", "heatmap_norm_range_launch_speed"),
            ("launch_angle", "heatmap_range_launch_angle", "heatmap_norm_range_launch_angle"),
            ("hit_distance_sc", "heatmap_range_hit_distance", "heatmap_norm_range_hit_distance"),
            ("spray_angle", "heatmap_range_spray_angle", "heatmap_norm_range_spray_angle"),
        ]
        norm_ranges: dict[str, list[float]] = {}
        for col_name, phys_attr, norm_attr in _heatmap_range_map:
            if col_name in reg_norm_stats:
                mean, std = reg_norm_stats[col_name]
                phys = getattr(model_cfg, phys_attr)
                norm = [(phys[0] - mean) / std, (phys[1] - mean) / std]
                setattr(model_cfg, norm_attr, norm)
                norm_ranges[col_name] = norm

        # 設定モード用: ターゲット名→正規化値域の dict を保存
        if model_cfg.heatmap_heads is not None:
            model_cfg.heatmap_norm_ranges = norm_ranges

    # === Physics Consistency Loss（正規化パラメータが必要なためここで構築）===
    physics_loss_fn = None
    if train_cfg.loss_weight_physics > 0:
        physics_loss_fn = PhysicsLoss(
            reg_norm_stats=reg_norm_stats,
            target_reg_columns=data_cfg.target_reg,
            margin=train_cfg.physics_margin_degrees,
        )
        # device への移動はモデル構築後に実施

    # === Dataset & DataLoader ===
    print("Building datasets...")
    ds_kwargs = dict(
        max_seq_len=model_cfg.pitch_seq_max_len,
        batter_hist_max_atbats=model_cfg.batter_hist_max_atbats,
        batter_hist_max_pitches=model_cfg.batter_hist_max_pitches,
    )
    train_ds = create_dataset(train_df, data_cfg, norm_stats, reg_norm_stats, **ds_kwargs)
    val_ds = create_dataset(val_df, data_cfg, norm_stats, reg_norm_stats, **ds_kwargs)
    del train_df, val_df  # メモリ解放

    use_persistent = train_cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

    # === モデル構築 ===
    model_cfg.num_reg_targets = len(data_cfg.target_reg)
    print("Building model...")
    model = build_model(data_cfg, model_cfg, stats)
    model = model.to(device)

    if physics_loss_fn is not None:
        physics_loss_fn = physics_loss_fn.to(device)

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
    loss_weight_str = (
        f"  Loss weights: SA={train_cfg.loss_weight_swing_attempt}, SR={train_cfg.loss_weight_swing_result}, "
        f"BT={train_cfg.loss_weight_bb_type}, Reg={train_cfg.loss_weight_regression}"
    )
    if train_cfg.loss_weight_physics > 0:
        loss_weight_str += f", Physics={train_cfg.loss_weight_physics}"
    print(loss_weight_str)

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
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            outputs = model_forward(model, batch, data_cfg, use_seq, use_batter_hist)
            loss, losses = compute_loss(outputs, batch, train_cfg, loss_fn_sr, loss_fn_bt, physics_loss_fn, model_cfg=model_cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_batches += 1

            postfix = {"total": f"{losses['total']:.4f}"}
            if model_scope in ("all", "swing_attempt", "classification"):
                postfix["SA"] = f"{losses.get('swing_attempt', 0.0):.4f}"
            if model_scope in ("all", "outcome", "classification"):
                postfix["SR"] = f"{losses.get('swing_result', 0.0):.4f}"
                postfix["BT"] = f"{losses.get('bb_type', 0.0):.4f}"
            if model_scope in ("all", "outcome", "regression"):
                postfix["Reg"] = f"{losses.get('regression', 0.0):.4f}"
            pbar.set_postfix(postfix)

        scheduler.step()

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # 検証
        val_metrics = evaluate(
            model,
            val_loader,
            train_cfg,
            data_cfg,
            device,
            model_scope,
            loss_fn_sr,
            loss_fn_bt,
            use_seq,
            use_batter_hist,
            physics_loss_fn,
            model_cfg=model_cfg,
        )

        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        record.update({f"train_{k}": v for k, v in avg_train.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)

        train_loss_str = _format_loss_parts(avg_train, model_scope)
        val_loss_str = _format_loss_parts(val_metrics, model_scope)

        acc_parts = []
        if "acc_swing_attempt" in val_metrics:
            acc_parts.append(f"Val SA Acc: {val_metrics['acc_swing_attempt']:.4f}")
        if "acc_swing_result" in val_metrics:
            acc_parts.append(f"Val SR Acc: {val_metrics['acc_swing_result']:.4f}")
        if "acc_bb_type" in val_metrics:
            acc_parts.append(f"Val BT Acc: {val_metrics['acc_bb_type']:.4f}")
        acc_str = " | ".join(acc_parts)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {avg_train['total']:.4f} ({train_loss_str}) | "
            f"Val Loss: {val_metrics['total']:.4f} ({val_loss_str})" + (f" | {acc_str}" if acc_str else "")
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
