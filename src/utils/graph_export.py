"""モデルの計算グラフを画像として出力するユーティリティ."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from config import ModelConfig


def create_dummy_inputs(
    model: nn.Module,
    model_cfg: ModelConfig,
    num_cont: int,
    num_ord: int,
    batch_size: int = 2,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """モデルの forward に渡すダミーテンソルを生成する."""
    # カテゴリカル特徴量
    cat_dict: dict[str, torch.Tensor] = {}
    # for feat_name, (num_classes, _embed_dim) in model_cfg.embedding_dims.items():
    for feat_name in model_cfg.embedding_dims:
        cat_dict[feat_name] = torch.zeros(batch_size, dtype=torch.long, device=device)

    # 連続値・順序特徴量
    cont = torch.zeros(batch_size, num_cont, device=device)
    ord_feat = torch.zeros(batch_size, num_ord, device=device)

    kwargs: dict[str, torch.Tensor] = {}

    # 系列モデル用のダミー入力
    is_seq = getattr(model, "is_seq_model", False)
    if is_seq and model_cfg.max_seq_len > 0:
        T = model_cfg.max_seq_len
        kwargs["seq_pitch_type"] = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        kwargs["seq_cont"] = torch.zeros(batch_size, T, num_cont, device=device)
        kwargs["seq_swing_attempt"] = torch.zeros(batch_size, T, device=device)
        kwargs["seq_swing_result"] = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        # マスクは全て有効にする（ゼロだと系列長0になりエンコーダが失敗するため）
        kwargs["seq_mask"] = torch.ones(batch_size, T, device=device)

    # 打者履歴モデル用のダミー入力
    is_batter_hist = getattr(model, "is_batter_hist_model", False)
    if is_batter_hist and model_cfg.batter_hist_max_atbats > 0:
        N = model_cfg.batter_hist_max_atbats
        P = model_cfg.batter_hist_max_pitches
        kwargs["hist_pitch_type"] = torch.zeros(batch_size, N, P, dtype=torch.long, device=device)
        kwargs["hist_cont"] = torch.zeros(batch_size, N, P, num_cont, device=device)
        kwargs["hist_swing_attempt"] = torch.zeros(batch_size, N, P, device=device)
        kwargs["hist_swing_result"] = torch.zeros(batch_size, N, P, dtype=torch.long, device=device)
        kwargs["hist_bb_type"] = torch.zeros(batch_size, N, dtype=torch.long, device=device)
        kwargs["hist_launch_speed"] = torch.zeros(batch_size, N, device=device)
        kwargs["hist_launch_angle"] = torch.zeros(batch_size, N, device=device)
        kwargs["hist_pitch_mask"] = torch.ones(batch_size, N, P, device=device)
        kwargs["hist_atbat_mask"] = torch.ones(batch_size, N, device=device)

    return cat_dict, cont, ord_feat, kwargs


def export_graph_torchview(
    model: nn.Module,
    cat_dict: dict[str, torch.Tensor],
    cont: torch.Tensor,
    ord_feat: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    output_path: Path,
    fmt: str = "png",
    depth: int = 3,
) -> Path:
    """torchview を使ってモデルグラフを保存する."""
    from torchview import draw_graph

    input_data = (cat_dict, cont, ord_feat)
    # draw_graph の **kwargs はそのまま model.forward() へ渡される
    graph = draw_graph(
        model,
        input_data=input_data,
        device=torch.device("cpu"),
        expand_nested=True,
        depth=depth,
        save_graph=False,
        graph_name=output_path.stem,
        **kwargs,
    )
    # graphviz Digraph を指定フォーマットでレンダリング
    dot = graph.visual_graph
    dot.format = fmt
    dot.render(output_path.stem, directory=str(output_path.parent), cleanup=True)
    return output_path.parent / f"{output_path.stem}.{fmt}"


def export_graph_torchviz(
    model: nn.Module,
    cat_dict: dict[str, torch.Tensor],
    cont: torch.Tensor,
    ord_feat: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    output_path: Path,
    fmt: str = "png",
) -> Path:
    """torchviz を使って autograd グラフを保存する."""
    from torchviz import make_dot

    outputs = model(cat_dict, cont, ord_feat, **kwargs)

    # regression が dict の場合（MDN モデル）は mu を使う
    reg = outputs["regression"]
    if isinstance(reg, dict):
        reg = reg["mu"]
        if reg.dim() == 3:
            reg = reg[:, 0, :]

    all_outputs = torch.cat(
        [
            outputs["swing_attempt"].unsqueeze(-1),
            outputs["swing_result"],
            outputs["bb_type"],
            reg,
        ],
        dim=-1,
    )
    dot = make_dot(all_outputs, params=dict(model.named_parameters()))
    dot.format = fmt
    dot.render(output_path.stem, directory=str(output_path.parent), cleanup=True)
    return output_path.parent / f"{output_path.stem}.{fmt}"


def export_graph(
    model: nn.Module,
    cat_dict: dict[str, torch.Tensor],
    cont: torch.Tensor,
    ord_feat: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    output_path: Path,
    backend: str = "torchview",
    fmt: str = "png",
    depth: int = 3,
) -> Path:
    """バックエンドを選択してグラフをエクスポートする."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "torchview":
        return export_graph_torchview(model, cat_dict, cont, ord_feat, kwargs, output_path, fmt, depth)
    elif backend == "torchviz":
        return export_graph_torchviz(model, cat_dict, cont, ord_feat, kwargs, output_path, fmt)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'torchview' or 'torchviz'.")
