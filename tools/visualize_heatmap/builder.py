"""ヒートマップ可視化ツールのコアロジック（データロード・推論・描画）."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.image import imread
from torch.utils.data import DataLoader, Subset

_UNITS: dict[str, str] = {
    "launch_speed": "mph",
    "launch_angle": "deg",
    "hit_distance_sc": "ft",
    "spray_angle": "deg",
}

_LEGACY_TARGET_REG: list[str] = [
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "spray_angle",
]


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _norm_range(saved_model_cfg: dict, target: str) -> tuple[float, float]:
    """ターゲットの正規化値域を model_config から取得する."""
    norm_ranges = saved_model_cfg.get("heatmap_norm_ranges") or {}
    if target in norm_ranges:
        r = norm_ranges[target]
        return (float(r[0]), float(r[1]))
    # レガシーフィールドへ fallback
    legacy_map = {
        "launch_speed": "heatmap_norm_range_launch_speed",
        "launch_angle": "heatmap_norm_range_launch_angle",
        "hit_distance_sc": "heatmap_norm_range_hit_distance",
        "spray_angle": "heatmap_norm_range_spray_angle",
    }
    key = legacy_map.get(target)
    if key and key in saved_model_cfg:
        r = saved_model_cfg[key]
        return (float(r[0]), float(r[1]))
    raise ValueError(f"No norm range found for target: {target!r}")


def _denorm_range(
    saved_model_cfg: dict,
    reg_norm_stats: dict[str, tuple[float, float]],
    target: str,
) -> tuple[float, float]:
    """正規化値域を物理値域に変換する."""
    norm_min, norm_max = _norm_range(saved_model_cfg, target)
    if target in reg_norm_stats:
        mean, std = reg_norm_stats[target]
        return (norm_min * std + mean, norm_max * std + mean)
    # norm_stats が無ければ heatmap_range_* を使う
    phys_key_map = {
        "launch_speed": "heatmap_range_launch_speed",
        "launch_angle": "heatmap_range_launch_angle",
        "hit_distance_sc": "heatmap_range_hit_distance",
        "spray_angle": "heatmap_range_spray_angle",
    }
    k = phys_key_map.get(target)
    if k and k in saved_model_cfg:
        r = saved_model_cfg[k]
        return (float(r[0]), float(r[1]))
    raise ValueError(f"Cannot determine physical range for target: {target!r}")


def _build_head_specs(saved_model_cfg: dict) -> list[dict]:
    """model_config からサブヘッド仕様リストを構築する（レガシー / 設定モード対応）."""
    heatmap_heads_raw = saved_model_cfg.get("heatmap_heads", None)

    if heatmap_heads_raw is None:
        # レガシーモード: ハードコード3サブヘッド
        return [
            {
                "type": "2d",
                "targets": ["launch_angle", "spray_angle"],
                "hm_key": "heatmap_2d",
                "off_key": "offset_2d",
                "grid_h": int(saved_model_cfg.get("heatmap_grid_h", 64)),
                "grid_w": int(saved_model_cfg.get("heatmap_grid_w", 64)),
            },
            {
                "type": "1d",
                "targets": ["launch_speed"],
                "hm_key": "heatmap_launch_speed",
                "off_key": "offset_launch_speed",
                "num_bins": int(saved_model_cfg.get("heatmap_num_bins", 64)),
            },
            {
                "type": "1d",
                "targets": ["hit_distance_sc"],
                "hm_key": "heatmap_hit_distance",
                "off_key": "offset_hit_distance",
                "num_bins": int(saved_model_cfg.get("heatmap_num_bins", 64)),
            },
        ]

    # 設定モード: heatmap_heads に基づく動的構築
    # make_heatmap_key は src/ に依存するため文字列で組み立てる
    def _make_key(htype: str, targets: list[str]) -> str:
        if htype == "2d":
            return f"2d_{'__'.join(targets)}"
        return f"1d_{targets[0]}"

    specs = []
    for hc in heatmap_heads_raw:
        htype = hc["type"]
        targets = hc["targets"]
        key = _make_key(htype, targets)
        spec: dict = {
            "type": htype,
            "targets": targets,
            "hm_key": f"heatmap_{key}",
            "off_key": f"offset_{key}",
        }
        if htype == "2d":
            spec["grid_h"] = int(hc.get("grid_h") or saved_model_cfg.get("heatmap_grid_h", 64))
            spec["grid_w"] = int(hc.get("grid_w") or saved_model_cfg.get("heatmap_grid_w", 64))
        else:
            spec["num_bins"] = int(hc.get("num_bins") or saved_model_cfg.get("heatmap_num_bins", 64))
        specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# データロード
# ---------------------------------------------------------------------------

def load_model_and_data(
    model_dir: Path,
    model_file: str,
    split: str,
    device: torch.device,
    src_dir: Path,
) -> tuple[nn.Module, object, dict, dict[str, tuple[float, float]], object]:
    """学習済みモデルとデータセットをロードする.

    Returns:
        (model, dataset, saved_model_cfg, reg_norm_stats, data_cfg)
    """
    import sys
    sys.path.insert(0, str(src_dir))

    from config import DataConfig, load_config
    from datasets import create_dataset, load_all_parquet_files, load_split_at_bat_ids
    from utils.model_io import load_trained_model

    # --- model_config.json ---
    model_config_path = model_dir / "model_config.json"
    with open(model_config_path) as f:
        saved_model_cfg: dict = json.load(f)

    # --- norm_params.json ---
    norm_params_path = model_dir / "norm_params.json"
    with open(norm_params_path) as f:
        norm_params = json.load(f)
    norm_stats: dict[str, tuple[float, float]] = {
        k: tuple(v) for k, v in norm_params["input"].items()
    }
    reg_norm_stats: dict[str, tuple[float, float]] = {
        k: tuple(v) for k, v in norm_params["target"].items()
    }

    # --- DataConfig ---
    config_yaml = model_dir / "config.yaml"
    if config_yaml.exists():
        data_cfg, _, _ = load_config(config_yaml)
    else:
        data_cfg = DataConfig()

    # --- モデルスコープ設定 ---
    model_scope: str = saved_model_cfg.get("model_scope", "all")
    max_seq_len: int = saved_model_cfg.get("pitch_seq_max_len", saved_model_cfg.get("max_seq_len", 0))
    use_batter_hist: bool = saved_model_cfg.get("batter_hist_max_atbats", 0) > 0
    batter_hist_max_atbats: int = saved_model_cfg.get("batter_hist_max_atbats", 0)
    batter_hist_max_pitches: int = saved_model_cfg.get("batter_hist_max_pitches", 10)
    need_at_bat_id = max_seq_len > 0 or use_batter_hist

    # --- データ読み込み ---
    print(f"Loading {split} data...")
    all_df = load_all_parquet_files(data_cfg.data_dir)
    split_ids = load_split_at_bat_ids(data_cfg.split_dir, split)

    if need_at_bat_id:
        df = all_df[all_df["at_bat_id"].isin(split_ids)].reset_index(drop=True)
    else:
        df = all_df[all_df["at_bat_id"].isin(split_ids)].drop(columns=["at_bat_id"]).reset_index(drop=True)
    del all_df
    print(f"  Samples: {len(df):,}")

    # outcome / regression スコープのフィルタ
    if model_scope == "outcome":
        df = df[df["swing_attempt"] == 1].reset_index(drop=True)
        print(f"  Filtered to swing_attempt=1: {len(df):,} samples")
    elif model_scope == "regression":
        if data_cfg.filter_swing_attempt:
            df = df[df["swing_attempt"] == 1].reset_index(drop=True)
            print(f"  Filtered to swing_attempt=1: {len(df):,} samples")
        if data_cfg.reg_target_filter in ("any", "all"):
            reg_cols = data_cfg.target_reg
            cond = (
                df[reg_cols].notna().any(axis=1)
                if data_cfg.reg_target_filter == "any"
                else df[reg_cols].notna().all(axis=1)
            )
            df = df[cond].reset_index(drop=True)
            print(f"  Filtered by reg_target_filter={data_cfg.reg_target_filter!r}: {len(df):,} samples")

    # --- データセット生成 ---
    dataset = create_dataset(
        df,
        data_cfg,
        norm_stats,
        reg_norm_stats,
        max_seq_len=max_seq_len,
        batter_hist_max_atbats=batter_hist_max_atbats,
        batter_hist_max_pitches=batter_hist_max_pitches,
    )

    # --- モデルロード ---
    model_path = model_dir / model_file
    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path, model_config_path, device)

    return model, dataset, saved_model_cfg, reg_norm_stats, data_cfg


# ---------------------------------------------------------------------------
# サンプル選択
# ---------------------------------------------------------------------------

def select_samples(
    dataset: object,
    num_samples: int,
    prefer_valid: bool,
    seed: int,
) -> list[int]:
    """データセットからサンプルインデックスを選択する.

    prefer_valid=True のとき、全回帰ターゲットのマスクが有効なサンプルを優先する。
    """
    rng = np.random.RandomState(seed)

    reg_mask_all = dataset.reg_mask.numpy()  # (N, D)
    all_valid = (reg_mask_all > 0.5).all(axis=1)  # (N,)
    n_total = len(all_valid)

    if prefer_valid:
        valid_idx = np.where(all_valid)[0]
        invalid_idx = np.where(~all_valid)[0]
        rng.shuffle(valid_idx)
        rng.shuffle(invalid_idx)
        combined = np.concatenate([valid_idx, invalid_idx])
    else:
        combined = np.arange(n_total)
        rng.shuffle(combined)

    selected = combined[:num_samples].tolist()
    n_valid = int(all_valid[selected].sum())
    print(f"  Selected {len(selected)} samples ({n_valid} with all reg targets valid)")
    return selected


# ---------------------------------------------------------------------------
# 推論
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_heatmap_outputs(
    model: nn.Module,
    dataset: object,
    indices: list[int],
    data_cfg: object,
    device: torch.device,
    saved_model_cfg: dict,
    batch_size: int = 256,
) -> list[dict]:
    """選択サンプルのヒートマップ出力を収集する.

    Returns:
        サンプルごとの dict リスト。各 dict は:
            - "heatmap_outputs": {key: numpy array}
            - "reg_targets": (D,) numpy
            - "reg_mask": (D,) numpy
    """
    from utils.inference import model_forward, move_batch_to_device

    use_seq = saved_model_cfg.get("pitch_seq_max_len", saved_model_cfg.get("max_seq_len", 0)) > 0
    use_batter_hist = saved_model_cfg.get("batter_hist_max_atbats", 0) > 0

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    results: list[dict] = []
    model.eval()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model_forward(model, batch, data_cfg, use_seq, use_batter_hist)

        reg_out = outputs.get("regression")
        if reg_out is None or not isinstance(reg_out, dict):
            raise ValueError(
                "モデルの出力に heatmap regression が含まれていません。"
                f"regression_head_type='heatmap' のモデルを指定してください。"
            )
        if not any(k.startswith("heatmap_") for k in reg_out):
            raise ValueError(
                "regression 出力にヒートマップキーが見つかりません。"
                "MLP/MDN モデルではなく heatmap モデルを指定してください。"
            )

        B = batch["reg_targets"].shape[0]
        for b in range(B):
            hm_dict = {k: v[b].cpu().numpy() for k, v in reg_out.items()}
            results.append({
                "heatmap_outputs": hm_dict,
                "reg_targets": batch["reg_targets"][b].cpu().numpy(),
                "reg_mask": batch["reg_mask"][b].cpu().numpy(),
            })

    return results


# ---------------------------------------------------------------------------
# 描画ヘルパー
# ---------------------------------------------------------------------------

def _plot_2d_panel(
    ax: plt.Axes,
    fig: plt.Figure,
    spec: dict,
    data: dict,
    saved_model_cfg: dict,
    reg_norm_stats: dict[str, tuple[float, float]],
    target_reg_list: list[str],
) -> None:
    """2D ヒートマップパネルを描画する."""
    import torch
    from models.components.heatmap_utils import decode_heatmap_2d

    hm = data["heatmap_outputs"][spec["hm_key"]]   # (1, H, W)
    off = data["heatmap_outputs"][spec["off_key"]]  # (2, H, W)
    hm_sq = hm[0]  # (H, W)

    t_h, t_w = spec["targets"]  # 行軸, 列軸
    grid_h = spec["grid_h"]
    grid_w = spec["grid_w"]

    # 物理値域
    phys_h_min, phys_h_max = _denorm_range(saved_model_cfg, reg_norm_stats, t_h)
    phys_w_min, phys_w_max = _denorm_range(saved_model_cfg, reg_norm_stats, t_w)

    # imshow: origin="lower" → y 軸は下が小、上が大
    im = ax.imshow(
        hm_sq,
        origin="lower",
        aspect="auto",
        extent=[phys_w_min, phys_w_max, phys_h_min, phys_h_max],
        cmap="hot",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 予測値のデコード
    norm_range_h = _norm_range(saved_model_cfg, t_h)
    norm_range_w = _norm_range(saved_model_cfg, t_w)
    pred_norm = decode_heatmap_2d(
        torch.tensor(hm[None], dtype=torch.float32),   # (1, 1, H, W)
        torch.tensor(off[None], dtype=torch.float32),  # (1, 2, H, W)
        value_range_h=norm_range_h,
        value_range_w=norm_range_w,
        grid_h=grid_h,
        grid_w=grid_w,
    )[0]  # (2,) 正規化値

    mean_h, std_h = reg_norm_stats.get(t_h, (0.0, 1.0))
    mean_w, std_w = reg_norm_stats.get(t_w, (0.0, 1.0))
    pred_phys_h = float(pred_norm[0].item()) * std_h + mean_h
    pred_phys_w = float(pred_norm[1].item()) * std_w + mean_w

    ax.plot(
        pred_phys_w, pred_phys_h, "bo",
        markersize=9, markeredgecolor="white", markeredgewidth=1.5,
        label=f"Pred ({pred_phys_w:.1f}, {pred_phys_h:.1f})",
        zorder=5,
    )

    # GT マーカー
    gt_label = "GT: N/A"
    if t_h in target_reg_list and t_w in target_reg_list:
        idx_h = target_reg_list.index(t_h)
        idx_w = target_reg_list.index(t_w)
        if data["reg_mask"][idx_h] > 0.5 and data["reg_mask"][idx_w] > 0.5:
            gt_norm_h = float(data["reg_targets"][idx_h])
            gt_norm_w = float(data["reg_targets"][idx_w])
            gt_phys_h = gt_norm_h * std_h + mean_h
            gt_phys_w = gt_norm_w * std_w + mean_w
            ax.plot(
                gt_phys_w, gt_phys_h, "rx",
                markersize=12, markeredgewidth=2.5,
                label=f"GT ({gt_phys_w:.1f}, {gt_phys_h:.1f})",
                zorder=6,
            )
            gt_label = f"GT ({gt_phys_w:.1f}, {gt_phys_h:.1f})"

    unit_h = _UNITS.get(t_h, "")
    unit_w = _UNITS.get(t_w, "")
    ax.set_xlabel(f"{t_w} [{unit_w}]", fontsize=9)
    ax.set_ylabel(f"{t_h} [{unit_h}]", fontsize=9)
    ax.set_title(
        f"{t_h} × {t_w}\n"
        f"Pred ({pred_phys_w:.1f}, {pred_phys_h:.1f}) | {gt_label}",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right")


def _plot_1d_panel(
    ax: plt.Axes,
    spec: dict,
    data: dict,
    saved_model_cfg: dict,
    reg_norm_stats: dict[str, tuple[float, float]],
    target_reg_list: list[str],
) -> None:
    """1D ヒートマップパネルを描画する."""
    import torch
    from models.components.heatmap_utils import decode_heatmap_1d

    hm = data["heatmap_outputs"][spec["hm_key"]]   # (1, L)
    off = data["heatmap_outputs"][spec["off_key"]]  # (1, L)
    hm_1d = hm[0]  # (L,)
    t = spec["targets"][0]
    num_bins = spec["num_bins"]

    norm_min, norm_max = _norm_range(saved_model_cfg, t)
    phys_min, phys_max = _denorm_range(saved_model_cfg, reg_norm_stats, t)
    mean, std = reg_norm_stats.get(t, (0.0, 1.0))

    # 各ビン中心の物理値
    bin_width_norm = (norm_max - norm_min) / num_bins
    bin_centers_norm = np.linspace(norm_min, norm_max, num_bins, endpoint=False) + bin_width_norm / 2
    bin_centers_phys = bin_centers_norm * std + mean
    bin_width_phys = (phys_max - phys_min) / num_bins

    ax.bar(
        bin_centers_phys, hm_1d,
        width=bin_width_phys * 0.9,
        color="steelblue", alpha=0.75,
        label="Heatmap",
    )

    # 予測値
    pred_norm_scalar = decode_heatmap_1d(
        torch.tensor(hm[None], dtype=torch.float32),   # (1, 1, L)
        torch.tensor(off[None], dtype=torch.float32),  # (1, 1, L)
        value_range=(norm_min, norm_max),
        num_bins=num_bins,
    )[0].item()
    pred_phys = float(pred_norm_scalar) * std + mean
    ax.axvline(pred_phys, color="blue", linewidth=2, label=f"Pred: {pred_phys:.1f}")

    # GT
    gt_label = "GT: N/A"
    if t in target_reg_list:
        idx = target_reg_list.index(t)
        if data["reg_mask"][idx] > 0.5:
            gt_norm_scalar = float(data["reg_targets"][idx])
            gt_phys = gt_norm_scalar * std + mean
            ax.axvline(gt_phys, color="red", linewidth=2, linestyle="--", label=f"GT: {gt_phys:.1f}")
            gt_label = f"GT: {gt_phys:.1f}"

    unit = _UNITS.get(t, "")
    ax.set_xlabel(f"{t} [{unit}]", fontsize=9)
    ax.set_ylabel("Heatmap response", fontsize=9)
    ax.set_title(f"{t}\nPred: {pred_phys:.1f} | {gt_label}", fontsize=9)
    ax.set_xlim(phys_min, phys_max)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# 図の描画・保存
# ---------------------------------------------------------------------------

def render_sample_figure(
    data: dict,
    saved_model_cfg: dict,
    reg_norm_stats: dict[str, tuple[float, float]],
    out_path: Path,
    sample_label: str = "",
) -> None:
    """1サンプル分の可視化図を描画して PNG に保存する."""
    head_specs = _build_head_specs(saved_model_cfg)
    target_reg_list: list[str] = saved_model_cfg.get("heatmap_target_reg") or _LEGACY_TARGET_REG

    n_panels = len(head_specs)
    width_ratios = [1.5 if hs["type"] == "2d" else 1.0 for hs in head_specs]
    fig_w = sum(width_ratios) * 4.2
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_w, 5.0),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    for i, spec in enumerate(head_specs):
        ax = axes[i]
        if spec["type"] == "2d":
            _plot_2d_panel(ax, fig, spec, data, saved_model_cfg, reg_norm_stats, target_reg_list)
        else:
            _plot_1d_panel(ax, spec, data, saved_model_cfg, reg_norm_stats, target_reg_list)

    title = f"Heatmap Regression Visualization"
    if sample_label:
        title = f"{title}  —  {sample_label}"
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_overview_grid(
    fig_paths: list[Path],
    out_path: Path,
    ncols: int = 4,
) -> None:
    """個別 PNG を格子状に並べた概要図を生成する."""
    n = len(fig_paths)
    ncols_actual = min(ncols, n)
    nrows = math.ceil(n / ncols_actual)

    fig, axes = plt.subplots(nrows, ncols_actual, figsize=(ncols_actual * 8, nrows * 5))
    axes_flat = np.array(axes).reshape(-1)

    for i, path in enumerate(fig_paths):
        img = imread(str(path))
        axes_flat[i].imshow(img)
        axes_flat[i].axis("off")
        axes_flat[i].set_title(path.stem, fontsize=7)

    for j in range(len(fig_paths), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
