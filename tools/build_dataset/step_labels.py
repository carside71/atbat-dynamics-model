"""Step 3: ラベル生成・エンコーディング — description解析 + カテゴリカルエンコード."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tools.build_dataset.columns import (
    CATEGORICAL_FEATURES,
    DESCRIPTION_MAP,
    SWING_RESULT_CLASSES,
)


@dataclass
class LabelsReport:
    """ラベル生成ステップのレポート."""

    swing_breakdown: dict[str, int] = field(default_factory=dict)
    swing_result_dist: dict[str, int] = field(default_factory=dict)
    bb_type_dist: dict[str, int] = field(default_factory=dict)
    unknown_descriptions: dict[str, int] = field(default_factory=dict)
    stats_tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def display(self) -> None:
        from IPython.display import display as ipy_display

        print("=== Step 3: Labels ===")

        # スイング内訳
        print("\n  スイング内訳:")
        for k, v in self.swing_breakdown.items():
            print(f"    {k}: {v:,}")

        # 不明な description
        if self.unknown_descriptions:
            print(f"\n  未知の description ({len(self.unknown_descriptions)} 種):")
            for k, v in sorted(self.unknown_descriptions.items(), key=lambda x: -x[1])[:10]:
                print(f"    {k}: {v:,}")

        # クラス分布
        print("\n  swing_result 分布:")
        ipy_display(pd.DataFrame(list(self.swing_result_dist.items()), columns=["class", "count"]))

        print("\n  bb_type 分布:")
        ipy_display(pd.DataFrame(list(self.bb_type_dist.items()), columns=["class", "count"]))


def _parse_description(df: pd.DataFrame) -> pd.DataFrame:
    """description カラムを swing_attempt, swing_result に変換."""
    swing_attempt = []
    swing_result = []

    for desc in df["description"]:
        if desc in DESCRIPTION_MAP:
            sa, sr = DESCRIPTION_MAP[desc]
            swing_attempt.append(sa)
            swing_result.append(sr)
        else:
            swing_attempt.append(False)
            swing_result.append(None)

    df["swing_attempt"] = pd.array(swing_attempt, dtype="boolean")
    df["swing_result"] = pd.array(swing_result, dtype="string")
    return df


def _encode_categorical(series: pd.Series, feature_name: str) -> tuple[pd.Series, pd.DataFrame]:
    """カテゴリカル値を頻度降順の整数ラベルにエンコードする.

    Returns:
        (エンコード済みSeries, stats DataFrame)
    """
    counts = series.value_counts(dropna=True).sort_values(ascending=False)
    mapping = {val: idx for idx, val in enumerate(counts.index)}

    encoded = series.map(mapping)
    # NaN を保持（Int64 nullable）
    encoded = encoded.astype("Int64")

    stats_df = pd.DataFrame(
        {
            "class_label": range(len(counts)),
            feature_name: counts.index,
            "count": counts.values,
        }
    )
    return encoded, stats_df


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelsReport]:
    """ラベル生成・エンコーディングを実行する.

    Args:
        df: step_features から受け取ったDataFrame

    Returns:
        (ラベル付きDataFrame, レポート)
    """
    report = LabelsReport()

    # description パース
    unknown = df[~df["description"].isin(DESCRIPTION_MAP)]["description"].value_counts()
    report.unknown_descriptions = unknown.to_dict()
    df = _parse_description(df)
    df = df.drop(columns=["description"])

    # スイング内訳
    sa_counts = df["swing_attempt"].value_counts(dropna=False)
    report.swing_breakdown = {
        "swing": int(sa_counts.get(True, 0)),
        "no_swing": int(sa_counts.get(False, 0)),
    }

    # swing_result エンコード（固定順序: foul=0, hit_into_play=1, miss=2）
    sr_mapping = {v: i for i, v in enumerate(SWING_RESULT_CLASSES)}
    sr_counts = df["swing_result"].value_counts(dropna=True)
    report.swing_result_dist = {cls: int(sr_counts.get(cls, 0)) for cls in SWING_RESULT_CLASSES}

    stats_tables = {}
    stats_tables["swing_result"] = pd.DataFrame(
        {
            "class_label": range(len(SWING_RESULT_CLASSES)),
            "swing_result": SWING_RESULT_CLASSES,
            "count": [int(sr_counts.get(cls, 0)) for cls in SWING_RESULT_CLASSES],
        }
    )
    df["swing_result"] = df["swing_result"].map(sr_mapping).astype("Int64")

    # swing_attempt を整数に
    df["swing_attempt"] = df["swing_attempt"].astype("Int64")

    # bb_type エンコード
    df["bb_type"], stats_tables["bb_type"] = _encode_categorical(df["bb_type"], "bb_type")
    report.bb_type_dist = dict(
        zip(
            stats_tables["bb_type"]["bb_type"],
            stats_tables["bb_type"]["count"],
            strict=True,
        )
    )

    # その他のカテゴリカル特徴量エンコード
    for feat in CATEGORICAL_FEATURES:
        if feat in df.columns:
            df[feat], stats_tables[feat] = _encode_categorical(df[feat], feat)

    # base_out_state, count_state の stats も生成
    for feat in ["base_out_state", "count_state"]:
        counts = df[feat].value_counts(dropna=True).sort_index()
        stats_tables[feat] = pd.DataFrame(
            {
                "class_label": counts.index,
                feat: counts.index,
                "count": counts.values,
            }
        )

    # stats_all 生成（全 stats を統合）
    all_rows = []
    for feat, st in stats_tables.items():
        for _, row in st.iterrows():
            all_rows.append(
                {
                    "feature": feat,
                    "class_label": row["class_label"],
                    "value": row[feat],
                    "count": row["count"],
                }
            )
    stats_tables["all"] = pd.DataFrame(all_rows)

    report.stats_tables = stats_tables
    return df, report


def save_stats(stats_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """stats CSVファイルを保存する."""
    for name, st in stats_tables.items():
        st.to_csv(output_dir / f"stats_{name}.csv", index=False)
