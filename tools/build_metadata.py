"""preprocess_01 と data parquet からビューア用メタデータを構築するスクリプト.

preprocess_01 と data parquet はファイル名・行数が 1:1 対応しているため、
行位置ベースで結合してメタデータテーブルを生成する。

Usage:
    python tools/build_metadata.py \
        --data-dir /workspace/datasets/statcast-customized/data \
        --preprocess-dir /workspace/datasets/statcast-customized-tmp/preprocess_01 \
        --output-dir /workspace/datasets/statcast-customized/metadata
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
            # API エラーは無視して続行
            pass
        if i + batch_size < len(ids_list):
            time.sleep(0.1)  # レートリミット対策

    return result


def build_metadata(data_dir: Path, preprocess_dir: Path, output_dir: Path, raw_csv_dir: Path | None = None) -> None:
    """メタデータファイルを構築する."""
    data_files = sorted(data_dir.glob("*.parquet"))
    preprocess_files = sorted(preprocess_dir.glob("*.parquet"))

    # ファイル名が一致するペアのみ処理
    p_map = {f.name: f for f in preprocess_files}
    pairs = [(d, p_map[d.name]) for d in data_files if d.name in p_map]
    print(f"Found {len(pairs)} matching file pairs")

    # --- per-sample metadata ---
    meta_rows = []
    # statcast CSV の player_name は **投手名** なので pitcher_mlbam → name にマッピング
    pitcher_name_map: dict[int, str] = {}
    all_batter_ids: set[int] = set()
    all_pitcher_ids: set[int] = set()

    for data_path, preprocess_path in tqdm(pairs, desc="Building metadata"):
        df_data = pd.read_parquet(data_path, columns=["at_bat_id", "batter", "game_pk"])
        df_p01 = pd.read_parquet(
            preprocess_path,
            columns=["batter", "pitcher", "player_name", "home_team", "away_team", "at_bat_number"],
        )

        assert len(df_data) == len(df_p01), (
            f"Row count mismatch: {data_path.name}: data={len(df_data)}, preprocess={len(df_p01)}"
        )

        # player_name は投手名（statcast の仕様）→ pitcher MLBAM ID → name
        for pitcher_id, name in zip(df_p01["pitcher"].values, df_p01["player_name"].values):
            if pd.notna(pitcher_id) and pd.notna(name):
                pitcher_name_map[int(pitcher_id)] = str(name)

        all_batter_ids.update(int(x) for x in df_p01["batter"].dropna().unique())
        all_pitcher_ids.update(int(x) for x in df_p01["pitcher"].dropna().unique())

        # at_bat_id 単位でメタデータを集約
        combined = pd.DataFrame(
            {
                "at_bat_id": df_data["at_bat_id"].values,
                "batter_class": df_data["batter"].values,
                "game_pk": df_data["game_pk"].values,
                "batter_mlbam": df_p01["batter"].values,
                "pitcher_mlbam": df_p01["pitcher"].values,
                "home_team": df_p01["home_team"].values,
                "away_team": df_p01["away_team"].values,
                "at_bat_number": df_p01["at_bat_number"].values,
            }
        )
        first_per_atbat = combined.groupby("at_bat_id").first().reset_index()
        meta_rows.append(first_per_atbat)

    # 元 CSV からも投手名を収集
    if raw_csv_dir and raw_csv_dir.exists():
        csv_files = sorted(raw_csv_dir.glob("statcast_*.csv"))
        print(f"Scanning {len(csv_files)} raw CSV files for pitcher names...")
        for csv_path in tqdm(csv_files, desc="Scanning CSVs"):
            df_csv = pd.read_csv(csv_path, usecols=["pitcher", "player_name"])
            for pitcher_id, name in zip(df_csv["pitcher"].values, df_csv["player_name"].values):
                if pd.notna(pitcher_id) and pd.notna(name):
                    pitcher_name_map[int(pitcher_id)] = str(name)

    meta_df = pd.concat(meta_rows, ignore_index=True)
    meta_df = meta_df.sort_values("at_bat_id").reset_index(drop=True)
    print(f"Total at-bats: {len(meta_df):,}")

    # 投手名カバレッジ
    pitchers_with_name = sum(1 for pid in all_pitcher_ids if pid in pitcher_name_map)
    print(f"Pitcher name coverage (from CSV): {pitchers_with_name}/{len(all_pitcher_ids)}")

    # --- MLB API から選手名を取得 ---
    # 打者名（player_name は投手名なので打者は別途取得が必要）
    # 投手で名前がないものも含めて一括取得
    player_name_map: dict[int, str] = dict(pitcher_name_map)  # まず投手名をコピー
    ids_to_fetch = set()
    # 打者 MLBAM ID で名前がないもの
    ids_to_fetch.update(all_batter_ids - set(player_name_map.keys()))
    # 投手で名前がないもの
    ids_to_fetch.update(all_pitcher_ids - set(player_name_map.keys()))

    if ids_to_fetch:
        print(f"Fetching {len(ids_to_fetch)} player names from MLB API...")
        api_names = _fetch_player_names_from_api(list(ids_to_fetch))
        player_name_map.update(api_names)
        print(f"  Fetched: {len(api_names)} names from API")

    # 最終カバレッジ
    batter_covered = sum(1 for bid in all_batter_ids if bid in player_name_map)
    pitcher_covered = sum(1 for pid in all_pitcher_ids if pid in player_name_map)
    print(
        f"Final coverage — Batters: {batter_covered}/{len(all_batter_ids)}, "
        f"Pitchers: {pitcher_covered}/{len(all_pitcher_ids)}"
    )

    # --- 保存 ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # at_bat_id → metadata parquet
    meta_path = output_dir / "atbat_metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)
    print(f"Saved at-bat metadata to {meta_path} ({len(meta_df):,} rows)")

    # MLBAM ID → player name JSON
    names_path = output_dir / "player_names.json"
    player_names_json = {str(k): v for k, v in sorted(player_name_map.items())}
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(player_names_json, f, ensure_ascii=False, indent=2)
    print(f"Saved player names to {names_path} ({len(player_names_json):,} entries)")


def main():
    parser = argparse.ArgumentParser(description="ビューア用メタデータ構築")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/datasets/statcast-customized/data",
        help="最終 data parquet ディレクトリ",
    )
    parser.add_argument(
        "--preprocess-dir",
        type=str,
        default="/workspace/datasets/statcast-customized-tmp/preprocess_01",
        help="preprocess_01 parquet ディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/datasets/statcast-customized/metadata",
        help="メタデータ出力ディレクトリ",
    )
    parser.add_argument(
        "--raw-csv-dir",
        type=str,
        default="/workspace/datasets/statcast",
        help="元の statcast CSV ディレクトリ（投手名カバレッジ向上用、省略可）",
    )
    args = parser.parse_args()

    raw_csv_dir = Path(args.raw_csv_dir) if args.raw_csv_dir else None
    build_metadata(Path(args.data_dir), Path(args.preprocess_dir), Path(args.output_dir), raw_csv_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
