"""データセット構築パイプラインのオーケストレータ."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.build_dataset import step_features, step_filter, step_labels, step_splits, step_validate


def run_pipeline(
    source_dir: str | Path = "/workspace/datasets/statcast",
    output_dir: str | Path = "/workspace/datasets/statcast-customized-v2",
    display: bool = True,
) -> pd.DataFrame:
    """データセット構築パイプラインを実行する.

    Args:
        source_dir: 生Statcast CSVファイルのディレクトリ
        output_dir: 出力ディレクトリ
        display: True の場合、各ステップの結果を表示する

    Returns:
        最終 DataFrame
    """
    # Step 1: Filter
    print("Step 1/5: Filtering batters...")
    df, filter_report = step_filter.run(str(source_dir))
    if display:
        filter_report.display()

    # Step 2: Feature Engineering
    print("\nStep 2/5: Engineering features...")
    df, features_report = step_features.run(df)
    if display:
        features_report.display()

    # Step 3: Labels
    print("\nStep 3/5: Generating labels...")
    df, labels_report = step_labels.run(df)
    if display:
        labels_report.display()

    # Step 4: Splits & Save
    print("\nStep 4/5: Splitting and saving...")
    df, splits_report = step_splits.run(df, labels_report.stats_tables, output_dir)
    if display:
        splits_report.display()

    # Step 5: Validate
    print("\nStep 5/5: Validating...")
    validate_report = step_validate.run(df)
    if display:
        validate_report.display()

    print(f"\nDone! Dataset saved to {output_dir}")
    return df
