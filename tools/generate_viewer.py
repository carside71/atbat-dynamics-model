"""テスト予測結果からインタラクティブ HTML ビューアを生成する CLI ツール."""

import argparse
from pathlib import Path

from viewer_builder import build_viewer_html, load_predictions, select_samples


def main():
    parser = argparse.ArgumentParser(description="AtBat Dynamics — サンプル単位の予測ビューア HTML を生成する")
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="predictions_*.npz と predictions_meta_*.json があるディレクトリ",
    )
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="HTML に含める最大サンプル数 (default: 2000)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="random",
        choices=["all", "misclassified_sa", "misclassified_sr", "misclassified_bt", "random", "include_invalid"],
        help="サンプル選択フィルタ (default: random, 欠損データは自動除外。include_invalid で含める)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="index",
        choices=["index", "sa_error", "reg_error"],
        help="ソート基準 (default: index)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力 HTML ファイルパス (default: pred-dir 内に viewer.html)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    template_path = Path(__file__).parent / "viewer_template.html"

    print(f"Loading predictions from {pred_dir} ...")
    preds, meta = load_predictions(pred_dir, args.split)
    n_total = len(preds["sa_prob"])
    print(f"  Total samples: {n_total:,}")

    print(f"Selecting samples (filter={args.filter}, sort={args.sort}, max={args.max_samples}) ...")
    indices = select_samples(
        preds,
        meta,
        max_samples=args.max_samples,
        filter_mode=args.filter,
        sort_by=args.sort,
        seed=args.seed,
    )
    print(f"  Selected: {len(indices):,} samples")

    print("Building HTML viewer ...")
    html = build_viewer_html(preds, meta, indices, template_path)

    output_path = Path(args.output) if args.output else pred_dir / "viewer.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Viewer saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
