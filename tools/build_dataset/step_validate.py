"""Step 5: GT整合性チェック."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tools.build_dataset.columns import (
    BB_TYPE_ANGLE_RANGES,
    COOCCUR_PAIRS,
    FAIL_THRESHOLD,
    # GT_CLS_COLS,
    GT_REG_COLS,
    PHYSICAL_BOUNDS,
)


@dataclass
class CheckResult:
    """個別チェックの結果."""

    name: str
    total: int
    violations: int
    detail: str = ""

    @property
    def rate(self) -> float:
        return self.violations / self.total if self.total > 0 else 0.0

    @property
    def status(self) -> str:
        if self.violations == 0:
            return "PASS"
        elif self.rate < FAIL_THRESHOLD:
            return "WARN"
        else:
            return "FAIL"


@dataclass
class ValidateReport:
    """バリデーションのレポート."""

    checks: list[CheckResult] = field(default_factory=list)

    def display(self) -> None:
        from IPython.display import display as ipy_display

        print("=== Step 5: Validate ===")

        status_icons = {"PASS": "[OK]", "WARN": "[!!]", "FAIL": "[NG]"}
        rows = []
        for c in self.checks:
            rows.append(
                {
                    "status": f"{status_icons[c.status]} {c.status}",
                    "check": c.name,
                    "violations": f"{c.violations:,}",
                    "rate": f"{c.rate:.4%}" if c.total > 0 else "-",
                    "detail": c.detail,
                }
            )
        ipy_display(pd.DataFrame(rows))

        fails = [c for c in self.checks if c.status == "FAIL"]
        if fails:
            print(f"\n  {len(fails)} 件のチェックが FAIL しました。")
        else:
            print("\n  全チェック PASS/WARN。")


def _check_hierarchical_null(df: pd.DataFrame) -> list[CheckResult]:
    """階層的NULL整合性チェック."""
    results = []
    # n = len(df) # not used, but can be used for debugging if needed

    # Rule A: swing_attempt=0 → swing_result, bb_type, REG_COLS は NaN
    no_swing = df["swing_attempt"] == 0
    dependent_cols = ["swing_result", "bb_type"] + GT_REG_COLS
    for col in dependent_cols:
        if col not in df.columns:
            continue
        violations = int((no_swing & df[col].notna()).sum())
        results.append(CheckResult(f"no_swing→{col}=NaN", int(no_swing.sum()), violations))

    # Rule B: swing_result≠1 → bb_type, hc_x, hc_y は NaN
    not_hit = df["swing_result"].notna() & (df["swing_result"] != 1)
    for col in ["bb_type", "hc_x", "hc_y"]:
        if col not in df.columns:
            continue
        violations = int((not_hit & df[col].notna()).sum())
        results.append(CheckResult(f"not_hit→{col}=NaN", int(not_hit.sum()), violations))

    # Rule C: swing_result=2 (miss) → REG_COLS は NaN
    miss = df["swing_result"] == 2
    for col in GT_REG_COLS:
        if col not in df.columns:
            continue
        violations = int((miss & df[col].notna()).sum())
        results.append(CheckResult(f"miss→{col}=NaN", int(miss.sum()), violations))

    return results


def _check_cooccurrence(df: pd.DataFrame) -> list[CheckResult]:
    """回帰ターゲットの共起チェック."""
    results = []
    for col_a, col_b in COOCCUR_PAIRS:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        a_na = df[col_a].isna()
        b_na = df[col_b].isna()
        violations = int((a_na != b_na).sum())
        results.append(CheckResult(f"cooccur({col_a},{col_b})", len(df), violations))
    return results


def _check_bb_type_angle(df: pd.DataFrame) -> list[CheckResult]:
    """bb_type vs launch_angle 整合性チェック."""
    results = []
    if "bb_type" not in df.columns or "launch_angle" not in df.columns:
        return results

    for bt_val, (lo, hi) in BB_TYPE_ANGLE_RANGES.items():
        mask = df["bb_type"] == bt_val
        subset = df.loc[mask, "launch_angle"].dropna()
        if len(subset) == 0:
            continue
        violations = int(((subset < lo) | (subset > hi)).sum())
        results.append(
            CheckResult(
                f"bb_type={bt_val} angle∈[{lo},{hi}]",
                len(subset),
                violations,
                f"range=[{subset.min():.1f}, {subset.max():.1f}]",
            )
        )
    return results


def _check_physical_bounds(df: pd.DataFrame) -> list[CheckResult]:
    """物理的境界チェック."""
    results = []
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        violations = int(((vals < lo) | (vals > hi)).sum())
        results.append(
            CheckResult(
                f"{col}∈[{lo},{hi}]",
                len(vals),
                violations,
                f"range=[{vals.min():.1f}, {vals.max():.1f}]",
            )
        )
    return results


def run(df: pd.DataFrame) -> ValidateReport:
    """GT整合性チェックを実行する.

    Args:
        df: 保存済みDataFrame

    Returns:
        バリデーションレポート
    """
    report = ValidateReport()

    report.checks.extend(_check_hierarchical_null(df))
    report.checks.extend(_check_cooccurrence(df))
    report.checks.extend(_check_bb_type_angle(df))
    report.checks.extend(_check_physical_bounds(df))

    return report
