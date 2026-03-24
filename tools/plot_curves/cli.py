"""学習曲線をプロットする CLI ツール.

使用例:
    # 全プロットを生成（デフォルト）
    python -m tools.plot_curves /workspace/outputs/all/dnn/2026-03-21-022236

    # プロット種類を指定
    python -m tools.plot_curves outputs/run1 --plots total_loss accuracy

    # 画像サイズ・フォントサイズを変更
    python -m tools.plot_curves outputs/run1 --figsize 16 10 --fontsize 14

    # PDF で出力
    python -m tools.plot_curves outputs/run1 --format pdf --dpi 300
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

ALL_PLOT_TYPES = ["total_loss", "individual_loss", "accuracy", "lr"]

# 個別損失コンポーネント（history.json 上のサフィックス）
_LOSS_COMPONENTS = ["swing_attempt", "swing_result", "bb_type", "regression", "physics"]

# 精度メトリクス（history.json 上のキー）
_ACC_KEYS = ["val_acc_swing_attempt", "val_acc_swing_result", "val_acc_bb_type"]

# 精度メトリクスの表示名
_ACC_LABELS = {
    "val_acc_swing_attempt": "Swing Attempt",
    "val_acc_swing_result": "Swing Result",
    "val_acc_bb_type": "Batted Ball Type",
}


def _load_history(output_dir: Path) -> list[dict]:
    """history.json を読み込む."""
    path = output_dir / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"history.json が見つかりません: {path}")
    with open(path) as f:
        return json.load(f)


def _extract(history: list[dict], key: str) -> list[float] | None:
    """history から指定キーの値リストを返す。キーが存在しなければ None."""
    values = [rec.get(key) for rec in history]
    if all(v is None for v in values):
        return None
    return [v if v is not None else float("nan") for v in values]


def plot_total_loss(history: list[dict], save_path: Path, figsize: tuple[float, float], dpi: int) -> None:
    """Total Loss の train/val 曲線."""
    epochs = [rec["epoch"] for rec in history]
    train = _extract(history, "train_total")
    val = _extract(history, "val_total")

    fig, ax = plt.subplots(figsize=figsize)
    if train:
        ax.plot(epochs, train, label="Train")
    if val:
        ax.plot(epochs, val, "--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_individual_loss(history: list[dict], save_path: Path, figsize: tuple[float, float], dpi: int) -> None:
    """個別損失コンポーネントのサブプロット."""
    epochs = [rec["epoch"] for rec in history]

    # 存在するコンポーネントのみ収集
    components = []
    for comp in _LOSS_COMPONENTS:
        train = _extract(history, f"train_{comp}")
        val = _extract(history, f"val_{comp}")
        if train or val:
            components.append((comp, train, val))

    if not components:
        print("  [SKIP] individual_loss: データなし")
        return

    n = len(components)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows / 2), squeeze=False)

    for idx, (comp, train, val) in enumerate(components):
        ax = axes[idx // cols][idx % cols]
        if train:
            ax.plot(epochs, train, label="Train")
        if val:
            ax.plot(epochs, val, "--", label="Val")
        ax.set_title(comp.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    # 余ったサブプロットを非表示
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Individual Losses", fontsize=plt.rcParams["font.size"] + 2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_accuracy(history: list[dict], save_path: Path, figsize: tuple[float, float], dpi: int) -> None:
    """分類タスクの精度曲線."""
    epochs = [rec["epoch"] for rec in history]

    has_data = False
    fig, ax = plt.subplots(figsize=figsize)
    for key in _ACC_KEYS:
        values = _extract(history, key)
        if values:
            ax.plot(epochs, values, label=_ACC_LABELS.get(key, key))
            has_data = True

    if not has_data:
        plt.close(fig)
        print("  [SKIP] accuracy: データなし")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_lr(history: list[dict], save_path: Path, figsize: tuple[float, float], dpi: int) -> None:
    """学習率の推移."""
    epochs = [rec["epoch"] for rec in history]
    lr = _extract(history, "lr")

    if not lr:
        print("  [SKIP] lr: データなし")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, lr)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# プロット種類 → (関数, ファイル名テンプレート)
_PLOT_REGISTRY: dict[str, tuple] = {
    "total_loss": (plot_total_loss, "loss_total"),
    "individual_loss": (plot_individual_loss, "loss_components"),
    "accuracy": (plot_accuracy, "accuracy"),
    "lr": (plot_lr, "lr"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="学習曲線をプロットする",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output_dir", type=str, help="train.py の出力ディレクトリ (history.json が存在するパス)")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=ALL_PLOT_TYPES,
        default=None,
        help=f"生成するプロットの種類 (default: 全種類). 選択肢: {', '.join(ALL_PLOT_TYPES)}",
    )
    parser.add_argument("--figsize", nargs=2, type=float, default=[12, 8], metavar=("W", "H"), help="画像サイズ (default: 12 8)")
    parser.add_argument("--fontsize", type=int, default=12, help="フォントサイズ (default: 12)")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", help="出力形式 (default: png)")
    parser.add_argument("--dpi", type=int, default=150, help="解像度 (default: 150)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    history = _load_history(output_dir)

    # 出力先ディレクトリ
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    fig_dir = output_dir / "figs" / timestamp
    fig_dir.mkdir(parents=True, exist_ok=True)

    # matplotlib 設定
    plt.rcParams.update({"font.size": args.fontsize})

    figsize = tuple(args.figsize)
    plot_types = args.plots or ALL_PLOT_TYPES

    print(f"Output dir: {output_dir}")
    print(f"Plots: {', '.join(plot_types)}")
    print(f"Save to: {fig_dir}")

    for ptype in plot_types:
        plot_fn, filename = _PLOT_REGISTRY[ptype]
        save_path = fig_dir / f"{filename}.{args.format}"
        plot_fn(history, save_path, figsize, args.dpi)

    print("完了しました。")


if __name__ == "__main__":
    main()
