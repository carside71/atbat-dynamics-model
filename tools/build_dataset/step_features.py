"""Step 2: 特徴量エンジニアリング — カラム選択・ゲームステート・軌道特徴量."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tools.build_dataset.columns import HC_X_CENTER, HC_Y_CENTER


@dataclass
class FeaturesReport:
    """特徴量エンジニアリングのレポート."""

    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    at_bat_count: int = 0

    def display(self) -> None:
        from IPython.display import display as ipy_display

        print("=== Step 2: Feature Engineering ===")
        print(f"  行数: {self.row_count:,}")
        print(f"  打席数: {self.at_bat_count:,}")
        print(f"  カラム数: {len(self.columns)}")

        # 欠損値テーブル
        nulls = {k: v for k, v in self.null_counts.items() if v > 0}
        if nulls:
            null_df = pd.DataFrame(
                [
                    {"column": k, "null_count": v, "null_pct": f"{v / self.row_count:.2%}"}
                    for k, v in sorted(nulls.items(), key=lambda x: -x[1])
                ]
            )
            print("\n  欠損値:")
            ipy_display(null_df)
        else:
            print("  欠損値: なし")

        print(f"\n  カラム一覧: {self.columns}")


def _assign_at_bat_id(df: pd.DataFrame) -> pd.DataFrame:
    """pitch_number == 1 を打席開始とし、at_bat_id を振る."""
    start_flag = (df["pitch_number"] == 1).astype(int)
    # 先頭行が打席開始でない場合の対応
    if start_flag.iloc[0] == 0:
        start_flag.iloc[0] = 1
    df.insert(0, "at_bat_id", start_flag.cumsum() - 1)
    return df


def _encode_base_out_state(df: pd.DataFrame) -> pd.Series:
    """outs × base runners → base_out_state (0-23)."""
    on_1b = df["on_1b"].notna().astype(int)
    on_2b = df["on_2b"].notna().astype(int)
    on_3b = df["on_3b"].notna().astype(int)
    base_state = on_1b + (on_2b * 2) + (on_3b * 4)
    outs = df["outs_when_up"].clip(0, 2)
    return (outs * 8 + base_state).astype("int8")


def _encode_count_state(df: pd.DataFrame) -> pd.Series:
    """balls × strikes → count_state (0-11)."""
    balls = df["balls"].clip(0, 3)
    strikes = df["strikes"].clip(0, 2)
    return (balls * 3 + strikes).astype("int8")


def _compute_spray_angle(df: pd.DataFrame) -> pd.Series:
    """hc_x, hc_y からスプレーアングルを算出（度）."""
    x = df["hc_x"] - HC_X_CENTER
    y = HC_Y_CENTER - df["hc_y"]
    return np.degrees(np.arctan2(x, y))


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, FeaturesReport]:
    """特徴量エンジニアリングを実行する.

    Args:
        df: step_filter から受け取った生DataFrame

    Returns:
        (加工済みDataFrame, レポート)
    """
    # ソート（ゲーム日時順→打席番号順で at_bat_id の一貫性を保証）
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    # at_bat_id 振り直し
    df = _assign_at_bat_id(df)

    # ゲームステート特徴量
    df["base_out_state"] = _encode_base_out_state(df)
    df["count_state"] = _encode_count_state(df)
    df["inning_clipped"] = df["inning"].clip(1, 10).astype("int16")
    df["is_inning_top"] = (df["inning_topbot"] == "Top").astype("int8")
    df["diff_score_clipped"] = (df["bat_score"] - df["fld_score"]).clip(-10, 10).astype("int8")
    df["pitch_number_clipped"] = df["pitch_number"].clip(1, 10).astype("int8")

    # plate_z 正規化
    sz_range = df["sz_top"] - df["sz_bot"]
    sz_range = sz_range.replace(0, np.nan)
    df["plate_z_norm"] = (df["plate_z"] - df["sz_bot"]) / sz_range

    # スプレーアングル
    df["spray_angle"] = _compute_spray_angle(df)

    # 最終カラム選択
    keep_cols = [
        # 識別子
        "at_bat_id",
        # ターゲット（生）
        "description",
        "bb_type",
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
        "hc_x",
        "hc_y",
        "spray_angle",
        # カテゴリカル入力
        "p_throws",
        "pitch_type",
        "batter",
        "stand",
        "base_out_state",
        "count_state",
        # 連続値入力
        "release_speed",
        "release_spin_rate",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "sz_top",
        "sz_bot",
        "plate_z_norm",
        # 順序入力
        "inning_clipped",
        "is_inning_top",
        "diff_score_clipped",
        "pitch_number_clipped",
        # ゲーム情報（分割・履歴用、最終保存時に使用）
        "game_pk",
        "game_date",
        # メタデータ用
        "pitcher",
        "home_team",
        "away_team",
        "at_bat_number",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # レポート
    report = FeaturesReport(
        columns=list(df.columns),
        row_count=len(df),
        null_counts={c: int(df[c].isna().sum()) for c in df.columns},
        at_bat_count=int(df["at_bat_id"].nunique()),
    )

    return df, report
