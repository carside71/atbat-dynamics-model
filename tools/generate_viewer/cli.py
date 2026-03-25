"""テスト予測結果からインタラクティブ HTML ビューアを生成する CLI ツール."""

import argparse
import sys
from pathlib import Path

from tools.generate_viewer.builder import (
    build_viewer_html,
    load_metadata,
    load_predictions,
    resolve_batter,
    select_samples,
)


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
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="/workspace/datasets/statcast-customized/metadata",
        help="atbat_metadata.parquet と player_names.json があるディレクトリ",
    )
    parser.add_argument(
        "--batter",
        type=str,
        default=None,
        help="特定の打者のデータのみ表示 (MLBAM ID または名前の一部, e.g. '608070' or 'Trout')",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    template_path = Path(__file__).parent / "viewer_template.html"

    print(f"Loading predictions from {pred_dir} ...")
    preds, meta = load_predictions(pred_dir, args.split)
    n_total = len(preds["sa_prob"])
    print(f"  Total samples: {n_total:,}")

    # メタデータ読み込み
    metadata_dir = Path(args.metadata_dir)
    atbat_meta = None
    player_names = None
    if metadata_dir.exists():
        print(f"Loading metadata from {metadata_dir} ...")
        atbat_meta, player_names = load_metadata(metadata_dir)
        print(f"  At-bat metadata: {len(atbat_meta):,} entries, Player names: {len(player_names):,} entries")
    else:
        print(f"  Metadata dir not found: {metadata_dir} (skipping metadata)")

    # 打者フィルタの解決
    batter_mlbam = None
    if args.batter:
        if atbat_meta is None or player_names is None:
            print("ERROR: --batter requires --metadata-dir with valid metadata files")
            sys.exit(1)
        if "meta_at_bat_id" not in preds:
            print("ERROR: --batter requires meta_at_bat_id in NPZ (re-run test.py with --save-predictions)")
            sys.exit(1)
        batter_mlbam = resolve_batter(args.batter, atbat_meta, player_names)
        if batter_mlbam is None:
            print(f"ERROR: Batter not found: '{args.batter}'")
            sys.exit(1)
        batter_name = player_names.get(str(batter_mlbam), str(batter_mlbam))
        print(f"  Batter filter: {batter_name} (MLBAM ID: {batter_mlbam})")

    filter_mode = args.filter
    max_samples = args.max_samples
    if batter_mlbam is not None:
        # 打者指定時: 全サンプルを表示するためデフォルトを調整
        if args.filter == "random":
            filter_mode = "all"
        if args.max_samples == 2000:
            max_samples = 100000  # 実質的に無制限

    print(f"Selecting samples (filter={filter_mode}, sort={args.sort}, max={max_samples}) ...")
    indices = select_samples(
        preds,
        meta,
        max_samples=max_samples,
        filter_mode=filter_mode,
        sort_by=args.sort,
        seed=args.seed,
        batter_mlbam=batter_mlbam,
        atbat_meta=atbat_meta,
    )
    print(f"  Selected: {len(indices):,} samples")

    if len(indices) == 0:
        print("WARNING: No samples selected. Check filter/batter options.")
        sys.exit(1)

    print("Building HTML viewer ...")
    html = build_viewer_html(preds, meta, indices, template_path, atbat_meta, player_names)

    output_path = Path(args.output) if args.output else pred_dir / "viewer.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Viewer saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
