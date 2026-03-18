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
    compute_embedding_dim,
    compute_normalization_stats,
    get_num_classes,
    load_parquet_files,
    load_stats,
)
from models import create_model


class TeeStream:
    """stdout/stderr を端末とファイルの両方に書き込むストリーム.

    tqdm 等の \r による上書き更新はファイル側ではバッファリングし、
    最終状態のみを書き込むことでログの肥大化を防ぐ。
    """

    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
        self._line_buf = ""
        self._closed = False

    def write(self, data):
        self.stream.write(data)
        if self._closed:
            return
        # ファイル側: \r を考慮してバッファリング
        self._line_buf += data
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            # \r がある場合は最後の \r 以降だけ残す（上書き表現）
            if "\r" in line:
                line = line.rsplit("\r", 1)[-1]
            self.file.write(line + "\n")
            self.file.flush()
        # バッファ中に \r があれば先頭まで巻き戻す
        if "\r" in self._line_buf:
            self._line_buf = self._line_buf.rsplit("\r", 1)[-1]

    def flush(self):
        self.stream.flush()
        if not self._closed:
            self.file.flush()

    def close_log(self):
        """残バッファをフラッシュしてファイルを閉じる."""
        if self._closed:
            return
        if self._line_buf.strip():
            if "\r" in self._line_buf:
                self._line_buf = self._line_buf.rsplit("\r", 1)[-1]
            self.file.write(self._line_buf + "\n")
        self._line_buf = ""
        self.file.flush()
        self.file.close()
        self._closed = True

    def isatty(self):
        return self.stream.isatty()


def build_model(data_cfg: DataConfig, model_cfg: ModelConfig, stats: dict) -> nn.Module:
    """»stats 情報から embedding_dims を設定してモデルを構築する."""
    num_classes = get_num_classes(stats)

    # 入力カテゴリカル特徴量のカーディナリティ
    cat_cardinality = {
        "p_throws": num_classes.get("p_throws", 2),
        "pitch_type": num_classes.get("pitch_type", 18),
        "batter": num_classes.get("batter", 783),
        "stand": num_classes.get("stand", 2),
        "base_out_state": 24,
        "count_state": 12,
    }

    model_cfg.embedding_dims = {
        feat: (cat_cardinality[feat], compute_embedding_dim(cat_cardinality[feat]))
        for feat in data_cfg.categorical_features
    }

    model_cfg.num_swing_result = num_classes.get("swing_result", 9)
    model_cfg.num_bb_type = num_classes.get("bb_type", 4)

    num_cont = len(data_cfg.continuous_features)
    num_ord = len(data_cfg.ordinal_features)

    return create_model(model_cfg.architecture, model_cfg, num_cont, num_ord)


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
    with open(output_dir / "train.log", "w") as log_file:
        sys.stdout = TeeStream(log_file, sys.__stdout__)
        sys.stderr = TeeStream(log_file, sys.__stderr__)
        try:
            _train(data_cfg, model_cfg, train_cfg, output_dir, log_file)
        finally:
            sys.stdout.close_log()
            sys.stderr.close_log()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


def _train(data_cfg, model_cfg, train_cfg, output_dir, log_file):
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # === データ読み込み ===
    print("Loading stats...")
    stats = load_stats(data_cfg.stats_dir)

    print("Loading training data...")
    train_df = load_parquet_files(data_cfg.data_dir, data_cfg.train_years)
    print(f"  Train samples: {len(train_df):,}")

    print("Loading validation data...")
    val_df = load_parquet_files(data_cfg.data_dir, data_cfg.val_years)
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
    model_info = {
        "architecture": model_cfg.architecture,
        "embedding_dims": model_cfg.embedding_dims,
        "backbone_hidden": model_cfg.backbone_hidden,
        "head_hidden": model_cfg.head_hidden,
        "dropout": model_cfg.dropout,
        "num_swing_result": model_cfg.num_swing_result,
        "num_bb_type": model_cfg.num_bb_type,
        "mdn_num_components": model_cfg.mdn_num_components,
        "num_cont": len(data_cfg.continuous_features),
        "num_ord": len(data_cfg.ordinal_features),
    }
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_info, f, indent=2)

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
