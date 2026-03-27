"""データセットの連続値カラムの分布を可視化する CLI ツール.

使用例:
    # 回帰ターゲットの 1D ヒストグラム（regression scope フィルタ付き）
    python -m tools.plot_distribution /workspace/datasets/statcast-customized-v2 \
        --columns launch_speed launch_angle hit_distance_sc spray_angle \
        --filter-swing --reg-target-filter all

    # 2D 密度プロット
    python -m tools.plot_distribution /workspace/datasets/statcast-customized-v2 \
        --plot-2d launch_angle:spray_angle launch_speed:hit_distance_sc \
        --filter-swing --reg-target-filter all

    # 投球特徴量の分布（フィルタなし）
    python -m tools.plot_distribution /workspace/datasets/statcast-customized-v2 \
        --columns release_speed release_spin_rate pfx_x pfx_z plate_x plate_z
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# src/ を import パスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.loaders import load_all_parquet_files, load_split_at_bat_ids

_DEFAULT_COLUMNS = ["launch_speed", "launch_angle", "hit_distance_sc", "spray_angle"]

_DEFAULT_REG_TARGETS = ["launch_speed", "launch_angle", "hit_distance_sc", "spray_angle"]


def _filter_dataframe(
    df: pd.DataFrame,
    filter_swing: bool,
    reg_target_filter: str,
    reg_targets: list[str],
) -> pd.DataFrame:
    """model_scope=regression 相当のフィルタリングを適用する."""
    n_before = len(df)

    if filter_swing:
        df = df[df["swing_attempt"] == 1].reset_index(drop=True)
        print(f"  swing_attempt==1 フィルタ: {n_before:,} -> {len(df):,}")

    if reg_target_filter in ("any", "all"):
        n_before_reg = len(df)
        if reg_target_filter == "any":
            mask = df[reg_targets].notna().any(axis=1)
        else:
            mask = df[reg_targets].notna().all(axis=1)
        df = df[mask].reset_index(drop=True)
        print(f"  reg_target_filter={reg_target_filter}: {n_before_reg:,} -> {len(df):,}")

    return df


def plot_hist(
    df: pd.DataFrame,
    column: str,
    save_path: Path,
    bins: int,
    figsize: tuple[float, float],
    dpi: int,
) -> None:
    """1D ヒストグラムを描画・保存する."""
    values = df[column].dropna()
    if values.empty:
        print(f"  [SKIP] {column}: データなし")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, edgecolor="white", linewidth=0.3, alpha=0.85)

    mean, std = values.mean(), values.std()
    vmin, vmax = values.min(), values.max()
    p1, p99 = np.percentile(values, [1, 99])

    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(
        f"{column}  (n={len(values):,})\n"
        f"mean={mean:.2f}  std={std:.2f}  "
        f"min={vmin:.2f}  max={vmax:.2f}  "
        f"p1={p1:.2f}  p99={p99:.2f}"
    )
    ax.grid(True, alpha=0.3)

    # p1/p99 の位置に縦線
    ax.axvline(p1, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label=f"p1={p1:.2f}")
    ax.axvline(p99, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label=f"p99={p99:.2f}")
    ax.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {save_path.name}  ({column}: mean={mean:.2f}, std={std:.2f}, range=[{vmin:.2f}, {vmax:.2f}])")


def plot_hist2d(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    save_path: Path,
    bins: int,
    figsize: tuple[float, float],
    dpi: int,
) -> None:
    """2D ヒストグラム（密度プロット）を描画・保存する."""
    mask = df[col_x].notna() & df[col_y].notna()
    x = df.loc[mask, col_x]
    y = df.loc[mask, col_y]

    if x.empty:
        print(f"  [SKIP] {col_x}:{col_y}: データなし")
        return

    fig, ax = plt.subplots(figsize=figsize)
    _, xedges, yedges, im = ax.hist2d(
        x,
        y,
        bins=bins,
        cmap="viridis",
        cmin=1,
    )

    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(f"{col_x} vs {col_y}  (n={len(x):,})")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] {save_path.name}  ({col_x} vs {col_y}, n={len(x):,})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="データセットの連続値カラムの分布を可視化する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset_dir", type=str, help="データセットディレクトリ（parquet + splits/）")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=f"1D ヒストグラムを描画するカラム (default: {', '.join(_DEFAULT_COLUMNS)})",
    )
    parser.add_argument(
        "--plot-2d",
        nargs="+",
        default=None,
        metavar="COL_X:COL_Y",
        help="2D 密度プロット。col_x:col_y 形式で複数指定可",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"], help="データ分割 (default: train)"
    )
    parser.add_argument("--filter-swing", action="store_true", help="swing_attempt==1 でフィルタ")
    parser.add_argument(
        "--reg-target-filter",
        default="none",
        choices=["none", "any", "all"],
        help="回帰ターゲットの欠損フィルタ (default: none)",
    )
    parser.add_argument(
        "--reg-targets",
        nargs="+",
        default=_DEFAULT_REG_TARGETS,
        help=f"--reg-target-filter の対象カラム (default: {', '.join(_DEFAULT_REG_TARGETS)})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/outputs/distribution",
        help="出力先ディレクトリ (default: /workspace/outputs/distribution)",
    )
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", help="出力形式 (default: png)")
    parser.add_argument("--dpi", type=int, default=150, help="解像度 (default: 150)")
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[8, 6], metavar=("W", "H"), help="画像サイズ (default: 8 6)"
    )
    parser.add_argument("--bins", type=int, default=100, help="ヒストグラムのビン数 (default: 100)")
    parser.add_argument("--fontsize", type=int, default=12, help="フォントサイズ (default: 12)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figsize = tuple(args.figsize)

    plt.rcParams.update({"font.size": args.fontsize})

    # データ読み込み
    print(f"Loading data from {dataset_dir} ...")
    df = load_all_parquet_files(dataset_dir)
    print(f"  全データ: {len(df):,} 行")

    # Split フィルタ
    split_dir = dataset_dir / "splits"
    if split_dir.exists():
        split_ids = load_split_at_bat_ids(split_dir, args.split)
        df = df[df["at_bat_id"].isin(split_ids)].reset_index(drop=True)
        print(f"  split={args.split}: {len(df):,} 行")
    else:
        print("  [WARN] splits/ が見つかりません。全データを使用します。")

    # Regression scope フィルタ
    df = _filter_dataframe(df, args.filter_swing, args.reg_target_filter, args.reg_targets)
    print(f"  最終サンプル数: {len(df):,} 行")

    # --columns 未指定 かつ --plot-2d も未指定 → デフォルトカラムで 1D
    # --columns 未指定 かつ --plot-2d 指定あり → 1D はスキップ
    columns = args.columns if args.columns is not None else ([] if args.plot_2d else _DEFAULT_COLUMNS)
    pairs_2d = args.plot_2d or []

    if not columns and not pairs_2d:
        print("描画対象が指定されていません。--columns または --plot-2d を指定してください。")
        return

    print(f"\nOutput dir: {output_dir}")

    # 1D ヒストグラム
    if columns:
        print(f"\n--- 1D Histograms ({len(columns)} columns) ---")
        for col in columns:
            if col not in df.columns:
                print(f"  [SKIP] {col}: カラムが存在しません")
                continue
            save_path = output_dir / f"hist_{col}.{args.format}"
            plot_hist(df, col, save_path, args.bins, figsize, args.dpi)

    # 2D 密度プロット
    if pairs_2d:
        print(f"\n--- 2D Density Plots ({len(pairs_2d)} pairs) ---")
        for pair in pairs_2d:
            if ":" not in pair:
                print(f"  [ERROR] 無効な形式: {pair} (col_x:col_y の形式で指定してください)")
                continue
            col_x, col_y = pair.split(":", 1)
            for c in (col_x, col_y):
                if c not in df.columns:
                    print(f"  [SKIP] {pair}: カラム {c} が存在しません")
                    break
            else:
                save_path = output_dir / f"hist2d_{col_x}_{col_y}.{args.format}"
                plot_hist2d(df, col_x, col_y, save_path, args.bins, figsize, args.dpi)

    print("\n完了しました。")


if __name__ == "__main__":
    main()
