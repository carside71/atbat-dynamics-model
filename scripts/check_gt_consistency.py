"""GT カラム間の整合性チェックツール.

データセットの GT カラム間に期待される論理的・物理的な整合性を検証し、
違反件数とサンプルをレポートファイルに出力する。

Usage:
    python scripts/check_gt_consistency.py
    python scripts/check_gt_consistency.py --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# src/ を import パスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from datasets.loaders import load_all_parquet_files

# === 定数 ===

# ホームプレート座標（画像座標系: 左上原点、y 軸下向き）
HOME_PLATE_X = 125.3
HOME_PLATE_Y = 167.8

# GT カラム名
CLS_COLS = ["swing_attempt", "swing_result", "bb_type"]
REG_COLS = ["launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y"]
ALL_GT_COLS = CLS_COLS + REG_COLS

# 回帰ターゲットの共起ペア
COOCCUR_PAIRS = [
    ("hc_x", "hc_y"),
    ("launch_speed", "launch_angle"),
]

# bb_type ごとの launch_angle 期待レンジ
BB_TYPE_NAMES = {0: "ground_ball", 1: "fly_ball", 2: "line_drive", 3: "popup"}
BB_TYPE_ANGLE_RANGES = {
    0: (-90.0, 80.0),   # ground_ball
    1: (10.0, 75.0),    # fly_ball
    2: (-15.0, 45.0),   # line_drive
    3: (20.0, 90.0),    # popup
}

# 物理的値域
PHYSICAL_BOUNDS = {
    "launch_speed": (0.0, 125.0),
    "launch_angle": (-90.0, 90.0),
    "hc_x": (0.0, 250.0),
    "hc_y": (0.0, 250.0),
}

# スプレーチャート距離と hit_distance_sc の相関閾値
SPRAY_CORR_THRESHOLD = 0.85

# ステータス判定の違反率閾値
FAIL_THRESHOLD = 0.001  # 0.1% 以上で FAIL


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────
def determine_status(violations: int, total: int) -> str:
    """違反件数から PASS / WARN / FAIL を判定する."""
    if violations == 0:
        return "PASS"
    rate = violations / total if total > 0 else 0.0
    return "FAIL" if rate >= FAIL_THRESHOLD else "WARN"


def status_icon(status: str) -> str:
    """ステータスに対応する視覚的アイコンを返す."""
    return {"PASS": "[OK]  ", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]


class ReportWriter:
    """レポートをバッファに書き込み、最後にファイルと stdout に出力する."""

    def __init__(self) -> None:
        self._buf = StringIO()

    def write(self, text: str = "") -> None:
        self._buf.write(text + "\n")

    def heading(self, title: str, level: int = 1) -> None:
        if level == 1:
            self.write("=" * 72)
            self.write(title)
            self.write("=" * 72)
        elif level == 2:
            self.write("")
            self.write("-" * 72)
            self.write(title)
            self.write("-" * 72)
        else:
            self.write(f"\n  {title}")

    def save(self, path: Path) -> None:
        """レポートをファイルに保存し、stdout にも出力する."""
        content = self._buf.getvalue()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(content, end="")
        print(f"\nReport saved to: {path}")


# ─────────────────────────────────────────────
# Check 0: サマリー統計
# ─────────────────────────────────────────────
def write_summary_statistics(w: ReportWriter, df: pd.DataFrame) -> None:
    """GT カラムの基本統計をレポートに書き込む."""
    total = len(df)
    w.heading("Overview", level=2)
    w.write(f"  Total rows: {total:,}")
    w.write("")

    # NULL 集計テーブル
    w.write(f"  {'Column':<20s} {'Non-Null':>12s} {'Null':>12s} {'Null%':>8s}")
    w.write("  " + "-" * 56)
    for col in ALL_GT_COLS:
        nn = int(df[col].notna().sum())
        na = int(df[col].isna().sum())
        pct = na / total * 100
        w.write(f"  {col:<20s} {nn:>12,} {na:>12,} {pct:>7.2f}%")

    # スイングカスケードの内訳
    n_no_swing = int((df["swing_attempt"] == 0).sum())
    n_swing = int((df["swing_attempt"] == 1).sum())
    sr_valid = df.loc[df["swing_attempt"] == 1, "swing_result"]
    n_foul = int((sr_valid == 0).sum())
    n_hip = int((sr_valid == 1).sum())
    n_miss = int((sr_valid == 2).sum())
    w.write("")
    w.write("  Swing cascade breakdown:")
    w.write(f"    No swing (swing_attempt=0):          {n_no_swing:>12,}")
    w.write(f"    Swing    (swing_attempt=1):           {n_swing:>12,}")
    w.write(f"      -> Foul        (swing_result=0):    {n_foul:>12,}")
    w.write(f"      -> Hit in play (swing_result=1):    {n_hip:>12,}")
    w.write(f"      -> Miss        (swing_result=2):    {n_miss:>12,}")


# ─────────────────────────────────────────────
# Check 1: 階層的 NULL 整合性
# ─────────────────────────────────────────────
def check_hierarchical_nulls(df: pd.DataFrame, max_examples: int) -> list[dict]:
    """スイングカスケードに基づく NULL 整合性を検証する."""
    results = []

    # Rule A: swing_attempt=0 → 下流カラムはすべて NaN
    no_swing = df[df["swing_attempt"] == 0]
    downstream_cols = ["swing_result", "bb_type"] + REG_COLS
    for col in downstream_cols:
        violations_idx = no_swing.index[no_swing[col].notna()]
        n_viol = len(violations_idx)
        results.append({
            "name": f"swing_attempt=0 -> {col} is NaN",
            "status": determine_status(n_viol, len(no_swing)),
            "violations": n_viol,
            "total": len(no_swing),
            "examples": violations_idx[:max_examples].tolist(),
        })

    # Rule B: swing_result != 1 (hit_into_play 以外) → bb_type, hc_x, hc_y は NaN
    # NOTE: launch_speed, launch_angle, hit_distance_sc はファウルでも記録される
    swing_non_hip = df[(df["swing_attempt"] == 1) & (df["swing_result"] != 1)]
    for col in ["bb_type", "hc_x", "hc_y"]:
        violations_idx = swing_non_hip.index[swing_non_hip[col].notna()]
        n_viol = len(violations_idx)
        results.append({
            "name": f"swing_result!=1 -> {col} is NaN",
            "status": determine_status(n_viol, len(swing_non_hip)),
            "violations": n_viol,
            "total": len(swing_non_hip),
            "examples": violations_idx[:max_examples].tolist(),
        })

    # Rule C: swing_result == 2 (miss) → 全回帰ターゲットが NaN
    swing_miss = df[(df["swing_attempt"] == 1) & (df["swing_result"] == 2)]
    for col in REG_COLS:
        violations_idx = swing_miss.index[swing_miss[col].notna()]
        n_viol = len(violations_idx)
        results.append({
            "name": f"swing_result=2(miss) -> {col} is NaN",
            "status": determine_status(n_viol, len(swing_miss)),
            "violations": n_viol,
            "total": len(swing_miss),
            "examples": violations_idx[:max_examples].tolist(),
        })

    return results


# ─────────────────────────────────────────────
# Check 2: 回帰ターゲットの共起性
# ─────────────────────────────────────────────
def check_regression_cooccurrence(df: pd.DataFrame, max_examples: int) -> list[dict]:
    """回帰ターゲットの共起パターンを検証する."""
    results = []

    for col_a, col_b in COOCCUR_PAIRS:
        a_present = df[col_a].notna()
        b_present = df[col_b].notna()
        # 片方だけ存在するケース
        mismatch = a_present != b_present
        violations_idx = df.index[mismatch]
        n_viol = len(violations_idx)
        total = len(df)
        results.append({
            "name": f"{col_a} <-> {col_b} co-occurrence",
            "status": determine_status(n_viol, total),
            "violations": n_viol,
            "total": total,
            "examples": violations_idx[:max_examples].tolist(),
        })

    return results


# ─────────────────────────────────────────────
# Check 3: スプレーチャート距離 vs hit_distance_sc
# ─────────────────────────────────────────────
def check_spray_distance_correlation(df: pd.DataFrame, max_examples: int) -> list[dict]:
    """hc_x, hc_y から算出したスプレー距離と hit_distance_sc の整合性を検証する."""
    mask = df["hc_x"].notna() & df["hc_y"].notna() & df["hit_distance_sc"].notna()
    sub = df.loc[mask].copy()
    total = len(sub)

    if total == 0:
        return [{
            "name": "spray_dist vs hit_distance_sc correlation",
            "status": "WARN",
            "violations": 0,
            "total": 0,
            "examples": [],
            "details": {},
        }]

    spray_dist = np.sqrt(
        (sub["hc_x"].to_numpy() - HOME_PLATE_X) ** 2
        + (sub["hc_y"].to_numpy() - HOME_PLATE_Y) ** 2
    )
    hit_dist = sub["hit_distance_sc"].to_numpy()

    # ピアソン相関
    corr = float(np.corrcoef(spray_dist, hit_dist)[0, 1])

    corr_status = "PASS" if corr >= SPRAY_CORR_THRESHOLD else "WARN"

    # ratio 外れ値検出（spray_dist > 1.0 のみ。極端に近い点は除外）
    valid = spray_dist > 1.0
    ratio = np.where(valid, hit_dist / spray_dist, np.nan)
    ratio_valid = ratio[~np.isnan(ratio)]
    q1, median, q3 = np.percentile(ratio_valid, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = median - 3 * iqr, median + 3 * iqr
    outlier_mask = valid & ((ratio < lo) | (ratio > hi))
    outlier_idx = sub.index[outlier_mask]
    n_outliers = len(outlier_idx)

    results = [
        {
            "name": "spray_dist vs hit_distance_sc correlation",
            "status": corr_status,
            "violations": 0 if corr_status == "PASS" else 1,
            "total": total,
            "examples": [],
            "details": {
                "n_samples": total,
                "pearson_r": round(corr, 6),
                "threshold": SPRAY_CORR_THRESHOLD,
            },
        },
        {
            "name": "spray_dist / hit_distance_sc ratio outliers",
            "status": determine_status(n_outliers, total),
            "violations": n_outliers,
            "total": total,
            "examples": outlier_idx[:max_examples].tolist(),
            "details": {
                "ratio_q1": round(float(q1), 4),
                "ratio_median": round(float(median), 4),
                "ratio_q3": round(float(q3), 4),
                "accepted_range": (round(float(lo), 4), round(float(hi), 4)),
            },
        },
    ]
    return results


# ─────────────────────────────────────────────
# Check 4: bb_type vs launch_angle
# ─────────────────────────────────────────────
def check_bb_type_launch_angle(df: pd.DataFrame, max_examples: int) -> list[dict]:
    """bb_type ごとの launch_angle 分布と期待レンジの整合性を検証する."""
    mask = df["bb_type"].notna() & df["launch_angle"].notna()
    sub = df.loc[mask]
    results = []

    for bt in sorted(BB_TYPE_ANGLE_RANGES.keys()):
        bt_mask = sub["bb_type"] == bt
        angles = sub.loc[bt_mask, "launch_angle"]
        if len(angles) == 0:
            continue

        lo, hi = BB_TYPE_ANGLE_RANGES[bt]
        oor = angles[(angles < lo) | (angles > hi)]
        oor_idx = oor.index
        name = BB_TYPE_NAMES[bt]

        results.append({
            "name": f"bb_type={bt}({name}) launch_angle in [{lo}, {hi}]",
            "status": determine_status(len(oor), len(angles)),
            "violations": len(oor),
            "total": len(angles),
            "examples": oor_idx[:max_examples].tolist(),
            "details": {
                "count": len(angles),
                "mean": round(float(angles.mean()), 1),
                "std": round(float(angles.std()), 1),
                "min": round(float(angles.min()), 1),
                "max": round(float(angles.max()), 1),
            },
        })

    return results


# ─────────────────────────────────────────────
# Check 5: 物理的値域チェック
# ─────────────────────────────────────────────
def check_physical_bounds(df: pd.DataFrame, max_examples: int) -> list[dict]:
    """回帰ターゲットが物理的に妥当な値域に収まるか検証する."""
    results = []

    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        valid = df[col].notna()
        vals = df.loc[valid, col]
        total = len(vals)
        if total == 0:
            results.append({
                "name": f"{col} in [{lo}, {hi}]",
                "status": "PASS",
                "violations": 0,
                "total": 0,
                "examples": [],
                "details": {},
            })
            continue

        oob = vals[(vals < lo) | (vals > hi)]
        results.append({
            "name": f"{col} in [{lo}, {hi}]",
            "status": determine_status(len(oob), total),
            "violations": len(oob),
            "total": total,
            "examples": oob.index[:max_examples].tolist(),
            "details": {
                "observed_min": round(float(vals.min()), 2),
                "observed_max": round(float(vals.max()), 2),
            },
        })

    return results


# ─────────────────────────────────────────────
# レポート出力
# ─────────────────────────────────────────────

# チェックごとの説明文
CHECK_DESCRIPTIONS = {
    "Check 1: Hierarchical Null Consistency": (
        "Verifies that downstream GT columns are NaN when upstream conditions\n"
        "  are not met, following the swing cascade logic:\n"
        "    - No swing (swing_attempt=0) -> all GT columns must be NaN\n"
        "    - Not hit into play (swing_result!=1) -> bb_type, hc_x, hc_y must be NaN\n"
        "    - Miss (swing_result=2) -> all regression targets must be NaN\n"
        "  Note: launch_speed, launch_angle, hit_distance_sc CAN exist for fouls."
    ),
    "Check 2: Regression Target Co-occurrence": (
        "Verifies that paired regression targets are both present or both missing.\n"
        "    - hc_x and hc_y should always co-occur\n"
        "    - launch_speed and launch_angle should always co-occur"
    ),
    "Check 3: Spray Distance vs hit_distance_sc": (
        "Compares the 2D spray chart distance from home plate (125.3, 167.8) with\n"
        "  the recorded hit_distance_sc. These differ because spray_dist is a 2D\n"
        "  top-down projection while hit_distance_sc is the 3D projected landing\n"
        "  distance. A Pearson correlation of ~0.90 is expected.\n"
        "  Ratio outliers are flagged using median +/- 3*IQR of (hit_distance_sc / spray_dist)."
    ),
    "Check 4: bb_type vs launch_angle": (
        "Verifies that launch_angle falls within expected ranges for each bb_type.\n"
        "  Statcast classifies bb_type using tracking data (trajectory shape, hang time),\n"
        "  not solely launch_angle, so some overlap at boundaries is normal.\n"
        "  Expected ranges:\n"
        "    - ground_ball: [-90, 80]   (typically negative angles)\n"
        "    - fly_ball:    [ 10, 75]   (mid-to-high angles)\n"
        "    - line_drive:  [-15, 45]   (low-to-mid angles)\n"
        "    - popup:       [ 20, 90]   (high angles)"
    ),
    "Check 5: Physical Bounds": (
        "Verifies that regression target values fall within physically plausible ranges.\n"
        "    - launch_speed:  [0, 125] mph\n"
        "    - launch_angle:  [-90, 90] degrees\n"
        "    - hc_x:          [0, 250] (spray chart pixel coords)\n"
        "    - hc_y:          [0, 250] (spray chart pixel coords)"
    ),
}


def write_check_section(
    w: ReportWriter, title: str, results: list[dict], max_examples: int,
) -> None:
    """チェック結果セクションをレポートに書き込む."""
    w.heading(title, level=2)

    # 説明文
    desc = CHECK_DESCRIPTIONS.get(title, "")
    if desc:
        w.write(f"  {desc}")
        w.write("")

    for r in results:
        icon = status_icon(r["status"])
        n = r["violations"]
        total = r["total"]
        pct = n / total * 100 if total > 0 else 0.0

        w.write(f"  {icon} {r['name']}")
        w.write(f"         Violations: {n:,} / {total:,} ({pct:.4f}%)")

        # details の内容を整形して出力
        details = r.get("details")
        if details and isinstance(details, dict):
            for k, v in details.items():
                if isinstance(v, tuple):
                    v = f"[{v[0]}, {v[1]}]"
                w.write(f"         {k}: {v}")

        # 違反サンプルの表示
        if n > 0 and r["examples"]:
            shown = r["examples"][:max_examples]
            w.write(f"         Example row indices (up to {max_examples}): {shown}")

        w.write("")


def write_final_summary(w: ReportWriter, all_results: list[dict]) -> None:
    """全チェック結果のサマリーをレポートに書き込む."""
    n_pass = sum(1 for r in all_results if r["status"] == "PASS")
    n_warn = sum(1 for r in all_results if r["status"] == "WARN")
    n_fail = sum(1 for r in all_results if r["status"] == "FAIL")
    total_violations = sum(r["violations"] for r in all_results)

    w.write("")
    w.heading("Final Summary", level=1)
    w.write("")
    w.write(f"  Total sub-checks:  {len(all_results)}")
    w.write(f"  [OK]   Passed:     {n_pass}")
    w.write(f"  [WARN] Warnings:   {n_warn}  (violations < 0.1% of checked rows)")
    w.write(f"  [FAIL] Failures:   {n_fail}  (violations >= 0.1% of checked rows)")
    w.write(f"  Total violations:  {total_violations:,}")
    w.write("")

    # WARN / FAIL の一覧を再掲
    flagged = [r for r in all_results if r["status"] != "PASS"]
    if flagged:
        w.write("  Flagged items:")
        for r in flagged:
            icon = status_icon(r["status"])
            n = r["violations"]
            total = r["total"]
            pct = n / total * 100 if total > 0 else 0.0
            w.write(f"    {icon} {r['name']}:  {n:,} / {total:,} ({pct:.4f}%)")
    else:
        w.write("  All checks passed. No issues found.")
    w.write("")


# ─────────────────────────────────────────────
# メインエントリポイント
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="GT column consistency checker")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/datasets/statcast-customized/data",
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/outputs",
        help="Directory to save the report file",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Max example row indices to show per violation",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    max_examples = args.max_examples

    w = ReportWriter()

    # ヘッダー
    w.heading("GT Column Consistency Report")
    w.write(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w.write(f"  Data dir: {data_dir}")
    w.write(f"  Status legend:")
    w.write(f"    [OK]   = 0 violations")
    w.write(f"    [WARN] = violations found, but < 0.1% of checked rows")
    w.write(f"    [FAIL] = violations >= 0.1% of checked rows")

    # データ読み込み
    print(f"Loading data from {data_dir} ...")
    df = load_all_parquet_files(data_dir)
    print(f"Loaded {len(df):,} rows. Running checks ...\n")

    all_results: list[dict] = []

    # Check 0: サマリー統計
    write_summary_statistics(w, df)

    # Check 1: 階層的 NULL 整合性
    results = check_hierarchical_nulls(df, max_examples)
    write_check_section(w, "Check 1: Hierarchical Null Consistency", results, max_examples)
    all_results.extend(results)

    # Check 2: 回帰ターゲットの共起性
    results = check_regression_cooccurrence(df, max_examples)
    write_check_section(w, "Check 2: Regression Target Co-occurrence", results, max_examples)
    all_results.extend(results)

    # Check 3: スプレーチャート距離 vs hit_distance_sc
    results = check_spray_distance_correlation(df, max_examples)
    write_check_section(w, "Check 3: Spray Distance vs hit_distance_sc", results, max_examples)
    all_results.extend(results)

    # Check 4: bb_type vs launch_angle
    results = check_bb_type_launch_angle(df, max_examples)
    write_check_section(w, "Check 4: bb_type vs launch_angle", results, max_examples)
    all_results.extend(results)

    # Check 5: 物理的値域
    results = check_physical_bounds(df, max_examples)
    write_check_section(w, "Check 5: Physical Bounds", results, max_examples)
    all_results.extend(results)

    # 最終サマリー
    write_final_summary(w, all_results)

    # ファイル出力
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"gt_consistency_report_{timestamp}.txt"
    w.save(output_path)

    # 終了コード
    has_fail = any(r["status"] == "FAIL" for r in all_results)
    sys.exit(1 if has_fail else 0)


if __name__ == "__main__":
    main()
