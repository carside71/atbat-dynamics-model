"""NPZ + メタ JSON を読み込み、自己完結型 HTML ビューアを生成するモジュール."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax


def load_predictions(pred_dir: str | Path, split: str = "test") -> tuple[dict[str, np.ndarray], dict]:
    """NPZ + メタ JSON を読み込む."""
    pred_dir = Path(pred_dir)
    npz = np.load(pred_dir / f"predictions_{split}.npz")
    with open(pred_dir / f"predictions_meta_{split}.json") as f:
        meta = json.load(f)
    return dict(npz), meta


def load_metadata(metadata_dir: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """メタデータディレクトリから打席メタデータと選手名辞書を読み込む.

    Returns:
        (atbat_meta_df, player_names)
        atbat_meta_df: at_bat_id → batter/pitcher/team 情報
        player_names: {mlbam_id_str: "Last, First"}
    """
    metadata_dir = Path(metadata_dir)
    atbat_meta = pd.read_parquet(metadata_dir / "atbat_metadata.parquet")
    with open(metadata_dir / "player_names.json", encoding="utf-8") as f:
        player_names = json.load(f)
    return atbat_meta, player_names


def _decode_base_out_state(val: int) -> str:
    """base_out_state (0-23) を人間可読な文字列に変換する."""
    outs = val // 8
    base_state = val % 8
    on_1b = base_state % 2
    on_2b = (base_state // 2) % 2
    on_3b = (base_state // 4) % 2
    runners = []
    if on_1b:
        runners.append("1B")
    if on_2b:
        runners.append("2B")
    if on_3b:
        runners.append("3B")
    runner_str = ",".join(runners) if runners else "---"
    return f"{runner_str} / {outs}out"


def _decode_count_state(val: int) -> str:
    """count_state (0-11) を人間可読な文字列に変換する."""
    balls = val // 3
    strikes = val % 3
    return f"{balls}-{strikes}"


def _is_valid_input(preds: dict[str, np.ndarray], idx: int) -> bool:
    """サンプルの入力特徴量が有効か判定する.

    連続値が全ゼロ（NaN が 0 に変換された）の行を無効とみなす。
    """
    if "cont" in preds:
        return not np.all(preds["cont"][idx] == 0.0)
    return True


def _build_sample_data(
    idx: int,
    preds: dict[str, np.ndarray],
    meta: dict,
    sample_metadata: dict | None = None,
) -> dict:
    """1 サンプル分のデータを辞書にまとめる."""
    valid_input = _is_valid_input(preds, idx)

    # --- 入力特徴量 ---
    inputs = {}

    # カテゴリカル特徴量
    cat_label_maps = meta.get("cat_label_maps", {})
    for col in meta["categorical_features"]:
        key = f"cat_{col}"
        if key in preds:
            raw_val = int(preds[key][idx])
            if raw_val < 0:
                inputs[col] = None
            elif col == "base_out_state":
                inputs[col] = _decode_base_out_state(raw_val)
            elif col == "count_state":
                inputs[col] = _decode_count_state(raw_val)
            elif col in cat_label_maps:
                inputs[col] = cat_label_maps[col].get(str(raw_val), str(raw_val))
            else:
                inputs[col] = str(raw_val)

    # 連続値特徴量（逆正規化して元スケールに戻す）
    if "cont" in preds:
        cont_vec = preds["cont"][idx]
        input_norm = meta.get("input_norm_stats", {})
        for i, col in enumerate(meta["continuous_features"]):
            if not valid_input:
                inputs[col] = None
            else:
                val = float(cont_vec[i])
                if col in input_norm:
                    mean, std = input_norm[col]
                    val = val * std + mean
                inputs[col] = round(val, 2)

    # 順序特徴量
    if "ord" in preds:
        ord_vec = preds["ord"][idx]
        for i, col in enumerate(meta["ordinal_features"]):
            if not valid_input:
                inputs[col] = None
            else:
                inputs[col] = int(ord_vec[i])

    # --- 予測 / GT ---
    # swing_attempt
    sa_prob = float(preds["sa_prob"][idx])
    sa_true = int(preds["sa_true"][idx])

    # swing_result (softmax 変換)
    sr_logits = preds["sr_logits"][idx].astype(float)
    sr_probs = softmax(sr_logits).tolist()
    sr_true = int(preds["sr_true"][idx])
    sr_names = meta.get("sr_names", [f"class_{i}" for i in range(len(sr_probs))])

    # bb_type (softmax 変換)
    bt_logits = preds["bt_logits"][idx].astype(float)
    bt_probs = softmax(bt_logits).tolist()
    bt_true = int(preds["bt_true"][idx])
    bt_names = meta.get("bt_names", [f"class_{i}" for i in range(len(bt_probs))])

    # regression (逆正規化)
    reg_cols = meta["reg_cols"]
    reg_norm = meta.get("reg_norm_stats", {})
    reg_pred_raw = preds["reg_pred"][idx]
    reg_true_raw = preds["reg_true"][idx]
    reg_mask = preds["reg_mask"][idx]
    regression = []
    for i, col in enumerate(reg_cols):
        valid = float(reg_mask[i]) > 0.5
        p = float(reg_pred_raw[i])
        t = float(reg_true_raw[i])
        if col in reg_norm:
            mean, std = reg_norm[col]
            p = p * std + mean
            t = t * std + mean
        regression.append(
            {
                "name": col,
                "pred": round(p, 2) if valid else None,
                "true": round(t, 2) if valid else None,
                "valid": valid,
            }
        )

    # ストライクゾーン用座標（逆正規化済みの plate_x, plate_z, sz_top, sz_bot を取得）
    strike_zone = None
    if "cont" in preds and valid_input:
        cont_names = meta["continuous_features"]
        cont_vec = preds["cont"][idx]
        input_norm = meta.get("input_norm_stats", {})

        def _get_denorm(col_name: str) -> float | None:
            if col_name not in cont_names:
                return None
            ci = cont_names.index(col_name)
            v = float(cont_vec[ci])
            if col_name in input_norm:
                m, s = input_norm[col_name]
                v = v * s + m
            return round(v, 3)

        px = _get_denorm("plate_x")
        pz = _get_denorm("plate_z")
        sz_t = _get_denorm("sz_top")
        sz_b = _get_denorm("sz_bot")
        if px is not None and pz is not None:
            strike_zone = {
                "plate_x": px,
                "plate_z": pz,
                "sz_top": sz_t if sz_t is not None else 3.5,
                "sz_bot": sz_b if sz_b is not None else 1.5,
            }

    return {
        "idx": idx,
        "valid_input": valid_input,
        "inputs": inputs,
        "game_info": sample_metadata,
        "swing_attempt": {"prob": round(sa_prob, 4), "true": sa_true},
        "swing_result": {"probs": [round(p, 4) for p in sr_probs], "names": sr_names, "true": sr_true},
        "bb_type": {"probs": [round(p, 4) for p in bt_probs], "names": bt_names, "true": bt_true},
        "regression": regression,
        "strike_zone": strike_zone,
    }


def select_samples(
    preds: dict[str, np.ndarray],
    meta: dict,
    max_samples: int = 2000,
    filter_mode: str = "all",
    sort_by: str = "index",
    seed: int = 42,
    batter_mlbam: int | None = None,
    atbat_meta: pd.DataFrame | None = None,
) -> list[int]:
    """表示するサンプルのインデックスを選択する."""
    n = len(preds["sa_prob"])
    indices = np.arange(n)

    # 欠損データ（cont 全ゼロ）を除外（valid_only フィルタ）
    if filter_mode != "include_invalid" and "cont" in preds:
        valid_mask = ~np.all(preds["cont"] == 0.0, axis=1)
        indices = indices[valid_mask]

    # 打者フィルタ（MLBAM ID 指定）
    if batter_mlbam is not None and "meta_at_bat_id" in preds and atbat_meta is not None:
        at_bat_ids = preds["meta_at_bat_id"]
        batter_atbat_ids = set(atbat_meta.loc[atbat_meta["batter_mlbam"] == batter_mlbam, "at_bat_id"].values)
        batter_mask = np.array([int(at_bat_ids[i]) in batter_atbat_ids for i in indices])
        indices = indices[batter_mask]

    if filter_mode == "misclassified_sa":
        sa_pred = (preds["sa_prob"] > 0.5).astype(int)
        sa_true = preds["sa_true"].astype(int)
        indices = indices[sa_pred != sa_true]
    elif filter_mode == "misclassified_sr":
        sr_true = preds["sr_true"]
        valid = sr_true >= 0
        sr_pred = preds["sr_logits"].argmax(axis=-1)
        mask = valid & (sr_pred != sr_true)
        indices = indices[mask]
    elif filter_mode == "misclassified_bt":
        bt_true = preds["bt_true"]
        valid = bt_true >= 0
        bt_pred = preds["bt_logits"].argmax(axis=-1)
        mask = valid & (bt_pred != bt_true)
        indices = indices[mask]
    elif filter_mode == "random":
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    # ソート
    if sort_by == "sa_error":
        sa_err = np.abs(preds["sa_prob"][indices] - preds["sa_true"][indices])
        order = np.argsort(-sa_err)  # 降順
        indices = indices[order]
    elif sort_by == "reg_error":
        mask = preds["reg_mask"][indices]
        pred_r = preds["reg_pred"][indices]
        true_r = preds["reg_true"][indices]
        err = np.where(mask > 0.5, (pred_r - true_r) ** 2, 0.0).sum(axis=-1)
        order = np.argsort(-err)
        indices = indices[order]

    if len(indices) > max_samples:
        indices = indices[:max_samples]

    return indices.tolist()


def build_sample_metadata(
    preds: dict[str, np.ndarray],
    sample_indices: list[int],
    atbat_meta: pd.DataFrame | None = None,
    player_names: dict[str, str] | None = None,
) -> dict[int, dict]:
    """各サンプルのゲームメタデータ辞書を構築する.

    Returns:
        {sample_idx: {batter_id, batter_name, pitcher_id, pitcher_name,
                       home_team, away_team, game_pk, game_date, at_bat_id, at_bat_number}}
    """
    result: dict[int, dict] = {}
    if atbat_meta is None or "meta_at_bat_id" not in preds:
        return result

    player_names = player_names or {}
    at_bat_ids = preds["meta_at_bat_id"]
    game_pks = preds.get("meta_game_pk")
    game_dates = preds.get("meta_game_date")

    # at_bat_id → メタデータの高速ルックアップ
    needed_ids = set(int(at_bat_ids[i]) for i in sample_indices)
    sub = atbat_meta[atbat_meta["at_bat_id"].isin(needed_ids)].set_index("at_bat_id")

    for idx in sample_indices:
        ab_id = int(at_bat_ids[idx])
        info: dict = {
            "at_bat_id": ab_id,
            "game_pk": int(game_pks[idx]) if game_pks is not None else None,
            "game_date": str(game_dates[idx]) if game_dates is not None else None,
        }
        if ab_id in sub.index:
            row = sub.loc[ab_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            batter_mlbam = int(row["batter_mlbam"]) if pd.notna(row.get("batter_mlbam")) else None
            pitcher_mlbam = int(row["pitcher_mlbam"]) if pd.notna(row.get("pitcher_mlbam")) else None
            info.update(
                {
                    "batter_id": batter_mlbam,
                    "batter_name": player_names.get(str(batter_mlbam), str(batter_mlbam)) if batter_mlbam else None,
                    "pitcher_id": pitcher_mlbam,
                    "pitcher_name": player_names.get(str(pitcher_mlbam), str(pitcher_mlbam)) if pitcher_mlbam else None,
                    "home_team": str(row["home_team"]) if pd.notna(row.get("home_team")) else None,
                    "away_team": str(row["away_team"]) if pd.notna(row.get("away_team")) else None,
                    "at_bat_number": int(row["at_bat_number"]) if pd.notna(row.get("at_bat_number")) else None,
                }
            )
        result[idx] = info
    return result


def resolve_batter(
    query: str,
    atbat_meta: pd.DataFrame,
    player_names: dict[str, str],
) -> int | None:
    """打者クエリ（MLBAM ID または名前の一部）を MLBAM ID に解決する."""
    # 数値ならMLBAM IDとして扱う
    try:
        mlbam_id = int(query)
        if mlbam_id in atbat_meta["batter_mlbam"].values:
            return mlbam_id
    except ValueError:
        pass

    # 名前で部分一致検索
    query_lower = query.lower()
    matches = []
    for id_str, name in player_names.items():
        if query_lower in name.lower():
            mlbam_id = int(id_str)
            if mlbam_id in atbat_meta["batter_mlbam"].values:
                matches.append((mlbam_id, name))

    if len(matches) == 1:
        return matches[0][0]
    elif len(matches) > 1:
        print("  複数の打者が一致しました:")
        for mid, mname in matches:
            count = (atbat_meta["batter_mlbam"] == mid).sum()
            print(f"    {mid}: {mname} ({count:,} at-bats)")
        print("  MLBAM ID を指定して再実行してください。")
        return None
    return None


def build_viewer_html(
    preds: dict[str, np.ndarray],
    meta: dict,
    sample_indices: list[int],
    template_path: str | Path,
    atbat_meta: pd.DataFrame | None = None,
    player_names: dict[str, str] | None = None,
) -> str:
    """サンプルデータを HTML テンプレートに埋め込んで完成した HTML を返す."""
    # ゲームメタデータの構築
    sample_meta = build_sample_metadata(preds, sample_indices, atbat_meta, player_names)

    samples = [_build_sample_data(i, preds, meta, sample_metadata=sample_meta.get(i)) for i in sample_indices]

    template_path = Path(template_path)
    template = template_path.read_text(encoding="utf-8")

    # JSON データをテンプレートに埋め込み
    samples_json = json.dumps(samples, ensure_ascii=False)
    html = template.replace("__SAMPLES_DATA__", samples_json)
    html = html.replace("__TOTAL_SAMPLES__", str(len(preds["sa_prob"])))
    html = html.replace("__SHOWN_SAMPLES__", str(len(samples)))

    return html
