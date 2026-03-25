"""ビューア用メタデータ（選手名）を構築するスクリプト.

パイプライン (build_dataset) が atbat_metadata.parquet を生成済みの前提で、
MLB Stats API から選手名を取得して player_names.json を生成する。

Usage:
    python -m tools.generate_viewer.metadata \
        --dataset-dir /workspace/datasets/statcast-customized-v2 \
        --raw-csv-dir /workspace/datasets/statcast
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

MLB_API_URL = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"


def _fetch_player_names_from_api(mlbam_ids: list[int], batch_size: int = 50) -> dict[int, str]:
    """MLB Stats API から選手名を一括取得する."""
    result: dict[int, str] = {}
    ids_list = list(mlbam_ids)

    for i in tqdm(range(0, len(ids_list), batch_size), desc="Fetching from MLB API"):
        batch = ids_list[i : i + batch_size]
        ids_str = ",".join(str(x) for x in batch)
        url = MLB_API_URL.format(ids=ids_str)
        try:
            with urlopen(url, timeout=10) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())
            for person in data.get("people", []):
                result[person["id"]] = person.get("lastFirstName", person.get("fullName", str(person["id"])))
        except (URLError, json.JSONDecodeError, KeyError):
            pass
        if i + batch_size < len(ids_list):
            time.sleep(0.1)

    return result


def build_player_names(dataset_dir: Path, raw_csv_dir: Path | None = None) -> None:
    """player_names.json を構築する."""
    meta_path = dataset_dir / "atbat_metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run build_dataset pipeline first.")

    meta_df = pd.read_parquet(meta_path)
    print(f"Loaded {len(meta_df):,} at-bat metadata rows")

    all_batter_ids = set(int(x) for x in meta_df["batter"].dropna().unique())
    all_pitcher_ids = set()
    if "pitcher" in meta_df.columns:
        all_pitcher_ids = set(int(x) for x in meta_df["pitcher"].dropna().unique())

    # 生CSVから投手名を収集
    pitcher_name_map: dict[int, str] = {}
    if raw_csv_dir and raw_csv_dir.exists():
        csv_files = sorted(raw_csv_dir.glob("statcast_*.csv"))
        print(f"Scanning {len(csv_files)} raw CSV files for pitcher names...")
        for csv_path in tqdm(csv_files, desc="Scanning CSVs"):
            df_csv = pd.read_csv(csv_path, usecols=["pitcher", "player_name"])
            for pitcher_id, name in zip(df_csv["pitcher"].values, df_csv["player_name"].values, strict=True):
                if pd.notna(pitcher_id) and pd.notna(name):
                    pitcher_name_map[int(pitcher_id)] = str(name)

    # API から不足分を取得
    player_name_map: dict[int, str] = dict(pitcher_name_map)
    ids_to_fetch = (all_batter_ids | all_pitcher_ids) - set(player_name_map.keys())

    if ids_to_fetch:
        print(f"Fetching {len(ids_to_fetch)} player names from MLB API...")
        api_names = _fetch_player_names_from_api(list(ids_to_fetch))
        player_name_map.update(api_names)
        print(f"  Fetched: {len(api_names)} names from API")

    # 保存
    names_path = dataset_dir / "player_names.json"
    player_names_json = {str(k): v for k, v in sorted(player_name_map.items())}
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(player_names_json, f, ensure_ascii=False, indent=2)
    print(f"Saved player names to {names_path} ({len(player_names_json):,} entries)")


def main():
    parser = argparse.ArgumentParser(description="ビューア用選手名メタデータ構築")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/workspace/datasets/statcast-customized-v2",
        help="データセットディレクトリ",
    )
    parser.add_argument(
        "--raw-csv-dir",
        type=str,
        default="/workspace/datasets/statcast",
        help="元の statcast CSV ディレクトリ（投手名カバレッジ向上用、省略可）",
    )
    args = parser.parse_args()

    raw_csv_dir = Path(args.raw_csv_dir) if args.raw_csv_dir else None
    build_player_names(Path(args.dataset_dir), raw_csv_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
