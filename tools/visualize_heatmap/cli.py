"""ヒートマップ可視化ツールの CLI エントリポイント."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """CLI メイン関数."""
    parser = argparse.ArgumentParser(
        description="ヒートマップ回帰モデルのテストデータに対して推論を行い、"
        "予測値・GT・ヒートマップを重ねて可視化する。",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="best_model.pt / model_config.json / norm_params.json を含むディレクトリ",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="best_model.pt",
        help="モデル重みファイル名（--model-dir 内、デフォルト: best_model.pt）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="可視化するデータ分割（デフォルト: test）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="生成するサンプル図の枚数（デフォルト: 10）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="PNG 出力先ディレクトリ（デフォルト: --model-dir/heatmap_vis/）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="サンプル選択の乱数シード（デフォルト: 42）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="推論バッチサイズ（デフォルト: 256）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="デバイス指定（例: cpu, cuda）。未指定時は自動検出",
    )
    parser.add_argument(
        "--no-prefer-valid",
        action="store_true",
        default=False,
        help="指定時: reg_mask が有効なサンプルの優先をしない（ランダム選択）",
    )
    parser.add_argument(
        "--overview-grid",
        action="store_true",
        default=False,
        help="指定時: 全サンプルを格子状に並べた概要 PNG を追加生成する",
    )
    args = parser.parse_args()

    import torch

    # src/ を Python パスに追加
    workspace_root = Path(__file__).resolve().parents[2]
    src_dir = workspace_root / "src"
    sys.path.insert(0, str(src_dir))

    from tools.visualize_heatmap.builder import (
        collect_heatmap_outputs,
        load_model_and_data,
        render_overview_grid,
        render_sample_figure,
        select_samples,
    )

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "heatmap_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    # --- モデル・データロード ---
    model, dataset, saved_model_cfg, reg_norm_stats, data_cfg = load_model_and_data(
        model_dir=model_dir,
        model_file=args.model_file,
        split=args.split,
        device=device,
        src_dir=src_dir,
    )

    # --- サンプル選択 ---
    prefer_valid = not args.no_prefer_valid
    indices = select_samples(
        dataset=dataset,
        num_samples=args.num_samples,
        prefer_valid=prefer_valid,
        seed=args.seed,
    )

    # --- 推論 ---
    print(f"Running inference on {len(indices)} samples...")
    sample_data = collect_heatmap_outputs(
        model=model,
        dataset=dataset,
        indices=indices,
        data_cfg=data_cfg,
        device=device,
        saved_model_cfg=saved_model_cfg,
        batch_size=args.batch_size,
    )

    # --- 描画・保存 ---
    fig_paths: list[Path] = []
    for i, (sample_idx, data) in enumerate(zip(indices, sample_data)):
        out_path = output_dir / f"sample_{i:04d}_idx{sample_idx}.png"
        render_sample_figure(
            data=data,
            saved_model_cfg=saved_model_cfg,
            reg_norm_stats=reg_norm_stats,
            out_path=out_path,
            sample_label=f"split={args.split}  dataset_idx={sample_idx}",
        )
        fig_paths.append(out_path)
        print(f"  [{i + 1}/{len(indices)}] Saved {out_path.name}")

    # --- 概要グリッド（オプション）---
    if args.overview_grid and fig_paths:
        grid_path = output_dir / "overview_grid.png"
        render_overview_grid(fig_paths=fig_paths, out_path=grid_path)
        print(f"Overview grid saved to {grid_path}")

    print(f"\nDone. {len(fig_paths)} figures saved to {output_dir}")
