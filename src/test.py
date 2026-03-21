"""学習済みモデルのテスト・性能評価スクリプト."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DataConfig, TrainConfig, load_config
from datasets import (
    StatcastBatterHistDataset,
    StatcastDataset,
    StatcastSequenceDataset,
    load_all_parquet_files,
    load_split_at_bat_ids,
    load_stats,
)
from utils.logging import tee_logging
from utils.model_io import load_trained_model


def _model_forward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    data_cfg: DataConfig,
    use_seq: bool,
    use_batter_hist: bool = False,
) -> dict[str, torch.Tensor]:
    """モデルの forward を呼び出す（シーケンス・打者履歴対応）."""
    cat_dict = {col: batch[col] for col in data_cfg.categorical_features}
    kwargs = {}
    if use_seq:
        kwargs.update(
            seq_pitch_type=batch["seq_pitch_type"],
            seq_cont=batch["seq_cont"],
            seq_swing_attempt=batch["seq_swing_attempt"],
            seq_swing_result=batch["seq_swing_result"],
            seq_mask=batch["seq_mask"],
        )
    if use_batter_hist:
        kwargs.update(
            hist_pitch_type=batch["hist_pitch_type"],
            hist_cont=batch["hist_cont"],
            hist_swing_attempt=batch["hist_swing_attempt"],
            hist_swing_result=batch["hist_swing_result"],
            hist_bb_type=batch["hist_bb_type"],
            hist_launch_speed=batch["hist_launch_speed"],
            hist_launch_angle=batch["hist_launch_angle"],
            hist_hc_x=batch["hist_hc_x"],
            hist_hc_y=batch["hist_hc_y"],
            hist_pitch_mask=batch["hist_pitch_mask"],
            hist_atbat_mask=batch["hist_atbat_mask"],
        )
    return model(cat_dict, batch["cont"], batch["ord"], **kwargs)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    data_cfg: DataConfig,
    device: torch.device,
    use_seq: bool = False,
    use_batter_hist: bool = False,
    save_inputs: bool = False,
) -> dict[str, np.ndarray]:
    """全バッチの予測とラベルを収集する."""
    all_sa_prob, all_sa_true = [], []
    all_sr_logits, all_sr_true = [], []
    all_bt_logits, all_bt_true = [], []
    all_reg_pred, all_reg_true, all_reg_mask = [], [], []

    # 入力特徴量の収集用（save_inputs=True 時のみ）
    all_cat: dict[str, list[np.ndarray]] = {col: [] for col in data_cfg.categorical_features} if save_inputs else {}
    all_cont: list[np.ndarray] = [] if save_inputs else []
    all_ord: list[np.ndarray] = [] if save_inputs else []

    for batch in tqdm(loader, desc="Predicting"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = _model_forward(model, batch, data_cfg, use_seq, use_batter_hist)

        all_sa_prob.append(outputs["swing_attempt"].sigmoid().cpu().numpy())
        all_sa_true.append(batch["swing_attempt"].cpu().numpy())

        all_sr_logits.append(outputs["swing_result"].cpu().numpy())
        all_sr_true.append(batch["swing_result"].cpu().numpy())

        all_bt_logits.append(outputs["bb_type"].cpu().numpy())
        all_bt_true.append(batch["bb_type"].cpu().numpy())

        reg_out = outputs["regression"]
        if isinstance(reg_out, dict):
            # MDN: 混合係数で重み付けした期待値を点推定とする
            pi = reg_out["pi"]  # (B, K)
            mu = reg_out["mu"]  # (B, K, D)
            # E[y] = sum_k pi_k * mu_k
            reg_pred = (pi.unsqueeze(-1) * mu).sum(dim=1)  # (B, D)
            all_reg_pred.append(reg_pred.cpu().numpy())
        else:
            all_reg_pred.append(reg_out.cpu().numpy())
        all_reg_true.append(batch["reg_targets"].cpu().numpy())
        all_reg_mask.append(batch["reg_mask"].cpu().numpy())

        if save_inputs:
            for col in data_cfg.categorical_features:
                all_cat[col].append(batch[col].cpu().numpy())
            all_cont.append(batch["cont"].cpu().numpy())
            all_ord.append(batch["ord"].cpu().numpy())

    result = {
        "sa_prob": np.concatenate(all_sa_prob),
        "sa_true": np.concatenate(all_sa_true),
        "sr_logits": np.concatenate(all_sr_logits),
        "sr_true": np.concatenate(all_sr_true),
        "bt_logits": np.concatenate(all_bt_logits),
        "bt_true": np.concatenate(all_bt_true),
        "reg_pred": np.concatenate(all_reg_pred),
        "reg_true": np.concatenate(all_reg_true),
        "reg_mask": np.concatenate(all_reg_mask),
    }

    if save_inputs:
        for col in data_cfg.categorical_features:
            result[f"cat_{col}"] = np.concatenate(all_cat[col])
        result["cont"] = np.concatenate(all_cont)
        result["ord"] = np.concatenate(all_ord)

    return result


def evaluate_swing_attempt(sa_prob: np.ndarray, sa_true: np.ndarray) -> dict:
    """swing_attempt の評価メトリクスを計算する."""
    sa_pred = (sa_prob > 0.5).astype(int)
    sa_true_int = sa_true.astype(int)

    metrics = {
        "accuracy": accuracy_score(sa_true_int, sa_pred),
        "f1": f1_score(sa_true_int, sa_pred),
        "roc_auc": roc_auc_score(sa_true_int, sa_prob),
    }

    cm = confusion_matrix(sa_true_int, sa_pred)
    report = classification_report(sa_true_int, sa_pred, target_names=["no_swing", "swing"], output_dict=True)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = report

    return metrics


def evaluate_multiclass(
    logits: np.ndarray,
    true: np.ndarray,
    class_names: list[str] | None = None,
    task_name: str = "",
) -> dict:
    """マルチクラス分類の評価メトリクスを計算する（有効サンプルのみ）."""
    mask = true >= 0
    if not mask.any():
        return {"n_samples": 0}

    logits = logits[mask]
    true = true[mask]
    pred = logits.argmax(axis=-1)

    metrics = {
        "n_samples": int(mask.sum()),
        "accuracy": accuracy_score(true, pred),
        "f1_macro": f1_score(true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(true, pred, average="weighted", zero_division=0),
    }

    labels = sorted(np.unique(np.concatenate([true, pred])))
    target_names = [class_names[i] if class_names and i < len(class_names) else str(i) for i in labels]
    report = classification_report(
        true, pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true, pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = report

    return metrics


def evaluate_regression(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    col_names: list[str],
    reg_norm_stats: dict[str, tuple[float, float]],
) -> dict:
    """回帰タスクの評価メトリクスを計算する（元スケールに逆変換）."""
    metrics = {}
    for i, col in enumerate(col_names):
        col_mask = mask[:, i] > 0.5
        if not col_mask.any():
            metrics[col] = {"n_samples": 0}
            continue

        p = pred[col_mask, i]
        t = true[col_mask, i]

        # 逆正規化
        if col in reg_norm_stats:
            mean, std = reg_norm_stats[col]
            p = p * std + mean
            t = t * std + mean

        metrics[col] = {
            "n_samples": int(col_mask.sum()),
            "mae": float(mean_absolute_error(t, p)),
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "r2": float(r2_score(t, p)),
        }

    return metrics


def print_results(results: dict) -> None:
    """テスト結果をフォーマットして表示する."""
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    # swing_attempt
    sa = results["swing_attempt"]
    print("\n--- swing_attempt (binary) ---")
    print(f"  Accuracy : {sa['accuracy']:.4f}")
    print(f"  F1       : {sa['f1']:.4f}")
    print(f"  ROC AUC  : {sa['roc_auc']:.4f}")
    cm = np.array(sa["confusion_matrix"])
    print("  Confusion Matrix:")
    print(f"    {'':>12s} pred_no  pred_yes")
    print(f"    {'true_no':>12s} {cm[0, 0]:>7d}  {cm[0, 1]:>8d}")
    print(f"    {'true_yes':>12s} {cm[1, 0]:>7d}  {cm[1, 1]:>8d}")

    # swing_result
    sr = results["swing_result"]
    print(f"\n--- swing_result (multi-class, n={sr['n_samples']:,}) ---")
    if sr["n_samples"] > 0:
        print(f"  Accuracy   : {sr['accuracy']:.4f}")
        print(f"  F1 (macro) : {sr['f1_macro']:.4f}")
        print(f"  F1 (weighted): {sr['f1_weighted']:.4f}")
        report = sr["classification_report"]
        print("  Per-class F1:")
        for k, v in report.items():
            if isinstance(v, dict) and "f1-score" in v:
                print(
                    f"    {k:>25s}: f1={v['f1-score']:.4f}  prec={v['precision']:.4f}  rec={v['recall']:.4f}  n={v['support']:.0f}"
                )

    # bb_type
    bt = results["bb_type"]
    print(f"\n--- bb_type (multi-class, n={bt['n_samples']:,}) ---")
    if bt["n_samples"] > 0:
        print(f"  Accuracy   : {bt['accuracy']:.4f}")
        print(f"  F1 (macro) : {bt['f1_macro']:.4f}")
        print(f"  F1 (weighted): {bt['f1_weighted']:.4f}")
        report = bt["classification_report"]
        print("  Per-class F1:")
        for k, v in report.items():
            if isinstance(v, dict) and "f1-score" in v:
                print(
                    f"    {k:>25s}: f1={v['f1-score']:.4f}  prec={v['precision']:.4f}  rec={v['recall']:.4f}  n={v['support']:.0f}"
                )

    # regression
    reg = results["regression"]
    print("\n--- regression (original scale) ---")
    for col, m in reg.items():
        if m["n_samples"] == 0:
            print(f"  {col}: no valid samples")
            continue
        print(f"  {col} (n={m['n_samples']:,}):")
        print(f"    MAE  : {m['mae']:.2f}")
        print(f"    RMSE : {m['rmse']:.2f}")
        print(f"    R²   : {m['r2']:.4f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AtBat Dynamics Model Test")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--model-dir", type=str, default=None, help="Directory containing model files (overrides config output_dir)"
    )
    parser.add_argument(
        "--model-file", type=str, default="best_model.pt", help="Model weights filename (default: best_model.pt)"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val"], help="Which split to evaluate (default: test)"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample predictions and inputs as NPZ for visualization",
    )
    args = parser.parse_args()

    if args.config:
        data_cfg, _, train_cfg = load_config(args.config)
    else:
        data_cfg = DataConfig()
        train_cfg = TrainConfig()

    model_dir = Path(args.model_dir) if args.model_dir else data_cfg.output_dir
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    # テスト出力ディレクトリを先に作成し、ログを端末とファイルの両方に出力
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    test_output_dir = model_dir / "test" / timestamp
    test_output_dir.mkdir(parents=True, exist_ok=True)

    with tee_logging(test_output_dir / "test.log"):
        _test(args, data_cfg, train_cfg, model_dir, test_output_dir, device)


def _test(args, data_cfg, train_cfg, model_dir, test_output_dir, device):
    print(f"Device: {device}")

    # === Stats 読み込み ===
    stats = load_stats(data_cfg.stats_dir)

    # ラベル名の取得
    sr_names = stats["swing_result"]["swing_result"].tolist() if "swing_result" in stats else None
    bt_names = stats["bb_type"]["bb_type"].tolist() if "bb_type" in stats else None

    # === 正規化パラメータ読み込み ===
    norm_params_path = model_dir / "norm_params.json"
    with open(norm_params_path) as f:
        norm_params = json.load(f)
    norm_stats = {k: tuple(v) for k, v in norm_params["input"].items()}
    reg_norm_stats = {k: tuple(v) for k, v in norm_params["target"].items()}

    # === モデル設定読み込み（シーケンス判定用） ===
    model_config_path = model_dir / "model_config.json"
    with open(model_config_path) as f:
        saved_model_cfg = json.load(f)
    use_seq = saved_model_cfg.get("max_seq_len", 0) > 0
    max_seq_len = saved_model_cfg.get("max_seq_len", 0)
    use_batter_hist = saved_model_cfg.get("batter_hist_max_atbats", 0) > 0
    batter_hist_max_atbats = saved_model_cfg.get("batter_hist_max_atbats", 0)
    batter_hist_max_pitches = saved_model_cfg.get("batter_hist_max_pitches", 10)
    need_at_bat_id = use_seq or use_batter_hist

    # === テストデータ読み込み ===
    print(f"Loading {args.split} data...")
    all_df = load_all_parquet_files(data_cfg.data_dir)
    split_ids = load_split_at_bat_ids(data_cfg.split_dir, args.split)

    if need_at_bat_id:
        test_df = all_df[all_df["at_bat_id"].isin(split_ids)].reset_index(drop=True)
    else:
        test_df = all_df[all_df["at_bat_id"].isin(split_ids)].drop(columns=["at_bat_id"]).reset_index(drop=True)
    del all_df
    print(f"  Samples: {len(test_df):,}")

    # メタデータ列を保存用に抽出（Dataset 構築前に取得）
    sample_meta_arrays = {}
    if args.save_predictions:
        sample_meta_arrays["meta_at_bat_id"] = (
            test_df["at_bat_id"].to_numpy(dtype=np.int64)
            if "at_bat_id" in test_df.columns
            else np.full(len(test_df), -1, dtype=np.int64)
        )
        sample_meta_arrays["meta_game_pk"] = (
            test_df["game_pk"].to_numpy(dtype=np.int64)
            if "game_pk" in test_df.columns
            else np.full(len(test_df), -1, dtype=np.int64)
        )
        if "game_date" in test_df.columns:
            sample_meta_arrays["meta_game_date"] = np.array([str(d) for d in test_df["game_date"]], dtype="U10")
        else:
            sample_meta_arrays["meta_game_date"] = np.full(len(test_df), "", dtype="U10")

    if use_batter_hist:
        test_ds = StatcastBatterHistDataset(
            test_df,
            data_cfg,
            max_seq_len,
            batter_hist_max_atbats,
            batter_hist_max_pitches,
            norm_stats,
            reg_norm_stats,
        )
    elif use_seq:
        test_ds = StatcastSequenceDataset(test_df, data_cfg, max_seq_len, norm_stats, reg_norm_stats)
    else:
        test_ds = StatcastDataset(test_df, data_cfg, norm_stats, reg_norm_stats)
    del test_df

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    # === モデル読み込み ===
    model_path = model_dir / args.model_file
    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path, model_config_path, device)

    # === 予測収集 ===
    preds = collect_predictions(model, test_loader, data_cfg, device, use_seq, use_batter_hist, save_inputs=args.save_predictions)

    # === 評価 ===
    results = {}
    results["swing_attempt"] = evaluate_swing_attempt(preds["sa_prob"], preds["sa_true"])
    results["swing_result"] = evaluate_multiclass(preds["sr_logits"], preds["sr_true"], sr_names, "swing_result")
    results["bb_type"] = evaluate_multiclass(preds["bt_logits"], preds["bt_true"], bt_names, "bb_type")
    results["regression"] = evaluate_regression(
        preds["reg_pred"],
        preds["reg_true"],
        preds["reg_mask"],
        data_cfg.target_reg,
        reg_norm_stats,
    )

    print_results(results)

    # === 結果を JSON で保存 ===
    output_path = test_output_dir / f"test_results_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # === 予測値・入力特徴量を NPZ で保存 ===
    if args.save_predictions:
        # サンプルメタデータを予測結果に追加
        preds.update(sample_meta_arrays)

        npz_path = test_output_dir / f"predictions_{args.split}.npz"
        np.savez_compressed(npz_path, **preds)
        print(f"Predictions saved to {npz_path}")

        # メタデータ JSON（ラベル名・正規化パラメータ等）
        cat_label_maps = {}
        for col in data_cfg.categorical_features:
            if col in stats:
                df_stat = stats[col]
                value_col = col  # stats CSV のカラム名は特徴量名と同じ
                cat_label_maps[col] = {int(row["class_label"]): str(row[value_col]) for _, row in df_stat.iterrows()}
        # stats_all.csv からの追加読み込み（base_out_state, count_state 等）
        if "all" in stats:
            stats_all_df = stats["all"]
            for col in data_cfg.categorical_features:
                if col not in cat_label_maps:
                    sub = stats_all_df[stats_all_df["feature"] == col]
                    if len(sub) > 0:
                        cat_label_maps[col] = {int(row["class_label"]): str(row["value"]) for _, row in sub.iterrows()}

        meta = {
            "sr_names": sr_names,
            "bt_names": bt_names,
            "reg_cols": list(data_cfg.target_reg),
            "reg_norm_stats": {k: list(v) for k, v in reg_norm_stats.items()},
            "input_norm_stats": {k: list(v) for k, v in norm_stats.items()},
            "continuous_features": list(data_cfg.continuous_features),
            "ordinal_features": list(data_cfg.ordinal_features),
            "categorical_features": list(data_cfg.categorical_features),
            "cat_label_maps": cat_label_maps,
        }
        meta_path = test_output_dir / f"predictions_meta_{args.split}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Predictions metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
