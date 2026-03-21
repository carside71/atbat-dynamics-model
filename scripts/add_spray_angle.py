"""spray_angle カラムを全 parquet ファイルに追加する.

hc_x, hc_y からスプレー角度を計算し、新カラムとして追加する。
hc_x/hc_y が NaN の行は spray_angle も NaN になる。

Usage:
    python scripts/add_spray_angle.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("/workspace/datasets/statcast-customized/data")

# ホームベースを原点とする座標変換の定数
HC_X_CENTER = 125.42
HC_Y_CENTER = 198.27


def add_spray_angle(df: pd.DataFrame) -> pd.DataFrame:
    """hc_x, hc_y から spray_angle (degrees) を計算して追加する."""
    x = df["hc_x"] - HC_X_CENTER
    y = HC_Y_CENTER - df["hc_y"]
    df["spray_angle"] = np.degrees(np.arctan2(x, y))
    return df


def main():
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {DATA_DIR}")

    total_rows = 0
    total_valid = 0

    for path in tqdm(parquet_files, desc="Adding spray_angle"):
        df = pd.read_parquet(path)
        add_spray_angle(df)

        valid = df["spray_angle"].notna().sum()
        total_rows += len(df)
        total_valid += valid

        df.to_parquet(path, index=False)

    print(f"Done. Total rows: {total_rows:,}, valid spray_angle: {total_valid:,}")


if __name__ == "__main__":
    main()
