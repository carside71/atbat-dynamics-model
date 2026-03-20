"""モデルのグラフ構造を画像として保存する CLI ツール.

使用例:
    # YAML 設定ファイルから単一モデルを出力
    python scripts/export_model_graph.py --config configs/dnn.yaml

    # アーキテクチャ名を直接指定
    python scripts/export_model_graph.py --arch atbat_resdnn --format pdf

    # 全登録モデルを一括出力
    python scripts/export_model_graph.py --all --output-dir outputs/graphs

    # torchviz バックエンドを使用
    python scripts/export_model_graph.py --config configs/dnn.yaml --backend torchviz

    # モジュール展開深度を指定 (torchview)
    python scripts/export_model_graph.py --arch atbat_seq_resdnn_batter_hist --depth 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# src ディレクトリをインポートパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DataConfig, ModelConfig, load_config
from datasets import compute_embedding_dim
from models import create_model
from utils.graph_export import create_dummy_inputs, export_graph

# --- デフォルトのカーディナリティ（stats 不要時のフォールバック） ---
_DEFAULT_CAT_CARDINALITY: dict[str, int] = {
    "p_throws": 2,
    "pitch_type": 18,
    "batter": 783,
    "stand": 2,
    "base_out_state": 24,
    "count_state": 12,
}

# --- アーキテクチャ名 → 代表 config YAML のマッピング ---
_NAME_TO_CONFIG: dict[str, str] = {
    "dnn": "configs/dnn.yaml",
    "dnn_mdn": "configs/dnn_mdn.yaml",
    "resdnn": "configs/resdnn.yaml",
    "resdnn_cascade": "configs/resdnn_cascade.yaml",
    "resdnn_focal": "configs/resdnn_focal.yaml",
    "seq_resdnn": "configs/seq_resdnn.yaml",
    "seq_resdnn_batter_hist": "configs/seq_resdnn_batter_hist.yaml",
}


def _build_model_from_config(data_cfg: DataConfig, model_cfg: ModelConfig) -> tuple[torch.nn.Module, int, int]:
    """config からモデルをビルドする（stats 不要のフォールバック対応）."""
    # embedding_dims が未設定ならデフォルトカーディナリティで埋める
    if not model_cfg.embedding_dims:
        model_cfg.embedding_dims = {
            feat: (
                _DEFAULT_CAT_CARDINALITY.get(feat, 10),
                compute_embedding_dim(_DEFAULT_CAT_CARDINALITY.get(feat, 10)),
            )
            for feat in data_cfg.categorical_features
        }

    num_cont = len(data_cfg.continuous_features)
    num_ord = len(data_cfg.ordinal_features)

    model = create_model(model_cfg, num_cont, num_ord)
    return model, num_cont, num_ord


def _try_load_config(config_path: str) -> tuple[DataConfig, ModelConfig]:
    """YAML から設定を読み込む."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent.parent / cfg_path
    data_cfg, model_cfg, _ = load_config(cfg_path)
    return data_cfg, model_cfg


def export_single_model(
    name: str,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    output_dir: Path,
    fmt: str,
    backend: str,
    depth: int,
    batch_size: int,
) -> None:
    """1 モデル分のグラフを生成・保存する."""
    model, num_cont, num_ord = _build_model_from_config(data_cfg, model_cfg)
    model.eval()

    cat_dict, cont, ord_feat, kwargs = create_dummy_inputs(
        model, model_cfg, num_cont, num_ord, batch_size=batch_size, device="cpu"
    )

    output_path = output_dir / f"{name}.{fmt}"
    result_path = export_graph(
        model, cat_dict, cont, ord_feat, kwargs, output_path, backend=backend, fmt=fmt, depth=depth
    )
    print(f"  [OK] {name} -> {result_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="モデルのグラフ構造を画像として保存する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="YAML 設定ファイルパス")
    group.add_argument("--name", type=str, help="プリセット名 (例: dnn, resdnn)")
    group.add_argument("--all", action="store_true", help="全プリセットを出力")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/outputs/graphs",
        help="保存先ディレクトリ (default: /workspace/outputs/graphs)",
    )
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", help="出力形式 (default: png)")
    parser.add_argument(
        "--backend",
        choices=["torchview", "torchviz"],
        default="torchview",
        help="描画バックエンド (default: torchview)",
    )
    parser.add_argument("--depth", type=int, default=3, help="モジュール展開深度 - torchview のみ (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2, help="ダミー入力のバッチサイズ (default: 2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {args.backend} | Format: {args.format} | Output: {output_dir}")

    if args.all:
        names = sorted(_NAME_TO_CONFIG.keys())
        print(f"全 {len(names)} モデルをエクスポートします: {', '.join(names)}")
        for name in names:
            data_cfg, model_cfg = _try_load_config(_NAME_TO_CONFIG[name])
            export_single_model(
                name, model_cfg, data_cfg, output_dir, args.format, args.backend, args.depth, args.batch_size
            )

    elif args.config:
        data_cfg, model_cfg = _try_load_config(args.config)
        name = Path(args.config).stem
        print(f"モデル '{name}' をエクスポートします (config: {args.config})")
        export_single_model(
            name, model_cfg, data_cfg, output_dir, args.format, args.backend, args.depth, args.batch_size
        )

    elif args.name:
        name = args.name
        config_yaml = _NAME_TO_CONFIG.get(name)
        if config_yaml is None:
            available = ", ".join(sorted(_NAME_TO_CONFIG.keys()))
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        data_cfg, model_cfg = _try_load_config(config_yaml)
        print(f"モデル '{name}' をエクスポートします")
        export_single_model(
            name, model_cfg, data_cfg, output_dir, args.format, args.backend, args.depth, args.batch_size
        )

    print("完了しました。")


if __name__ == "__main__":
    main()
