#!/usr/bin/env python3
"""
experiment/hipotesis_tests/variant_hypothesis_tests.py

Performs variant-specific paired hypothesis tests between CP and MILP median times
- Filters out rows where BOTH cp_median_time >= 600 AND milp_median_time >= 600
- For each specified variant runs Shapiro-Wilk on paired differences and a
  Wilcoxon signed-rank test with the requested alternative hypothesis.

Saves a text report at experiment/hipotesis_tests/variant_tests_results.txt
"""
from pathlib import Path
import sys
import math

import pandas as pd
from scipy import stats

# Configuration
ROOT = Path.cwd()
TABLE_CSV = ROOT / "experiment" / "tables" / "instances_solved_table.csv"
OUT_DIR = ROOT / "experiment" / "hipotesis_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "variant_tests_results.txt"

CP_COL = "cp_median_time"
MILP_COL = "milp_median_time"
VARIANTS_TO_TEST = [
    ("drc_jssp", "less"),     # one-tailed: CP faster (cp - milp < 0)
    ("sdst_jssp", "less"),    # one-tailed: CP faster
    ("f_jssp", "greater"),    # one-tailed: MILP faster (cp - milp > 0 => CP slower)
    ("basic_jssp", "two-sided"),
    ("d_jssp", "two-sided"),
    ("r_jssp", "two-sided"),
    ("t_jssp", "two-sided")
]
ALPHA = 0.05


def load_and_prepare(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # Ensure necessary columns exist
    for c in (CP_COL, MILP_COL, "variant"):
        if c not in df.columns:
            raise KeyError(f"Required column missing from CSV: {c}")

    # Coerce to numeric and drop rows with missing values in the two time cols
    df[CP_COL] = pd.to_numeric(df[CP_COL], errors="coerce")
    df[MILP_COL] = pd.to_numeric(df[MILP_COL], errors="coerce")
    df = df.dropna(subset=[CP_COL, MILP_COL]).reset_index(drop=True)

    # Filter out rows where BOTH times are >= 600 (censored ties)
    mask_both_censored = (df[CP_COL] >= 600) & (df[MILP_COL] >= 600)
    df_filtered = df[~mask_both_censored].copy()

    return df_filtered


def safe_shapiro(diffs):
    # Shapiro requires 3 <= n <= 5000
    n = len(diffs)
    if n < 3:
        return None, None, "insufficient_sample"
    if n > 5000:
        return None, None, "too_large_for_shapiro"
    try:
        stat, p = stats.shapiro(diffs)
        return float(stat), float(p), None
    except Exception as e:
        return None, None, str(e)


def safe_wilcoxon(cp_vals, milp_vals, alternative):
    # Wilcoxon requires at least one non-zero difference; SciPy may raise on degenerate data
    try:
        res = stats.wilcoxon(cp_vals, milp_vals, alternative=alternative, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue), None
    except Exception as e:
        return None, None, str(e)


def interpret_wilcoxon_p(p, alt):
    if p is None:
        return "Test could not be performed"
    reject = p < ALPHA
    if alt == 'less':
        if reject:
            return f"Reject H0 at alpha={ALPHA}: evidence that CP is faster (CP < MILP)"
        else:
            return f"Fail to reject H0 at alpha={ALPHA}: no evidence that CP is faster"
    if alt == 'greater':
        if reject:
            return f"Reject H0 at alpha={ALPHA}: evidence that CP is slower (MILP faster)"
        else:
            return f"Fail to reject H0 at alpha={ALPHA}: no evidence that MILP is faster"
    # two-sided
    if reject:
        return f"Reject H0 at alpha={ALPHA}: evidence of a difference between CP and MILP"
    return f"Fail to reject H0 at alpha={ALPHA}: no evidence of a difference"


def run_variant_tests(df, variants):
    lines = []
    lines.append("Variant-specific paired tests (after filtering censored ties)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total rows after filtering: {len(df)}")
    lines.append("Note: rows where BOTH cp_median_time >= 600 AND milp_median_time >= 600 were removed.")
    lines.append("")

    for variant, alt in variants:
        lines.append("-" * 70)
        lines.append(f"Variant: {variant}")
        dfv = df[df['variant'] == variant]
        n = len(dfv)
        lines.append(f"Sample size (pairs) after filtering: {n}")

        if n == 0:
            lines.append("No data available for this variant after filtering.\n")
            continue

        cp_vals = dfv[CP_COL].to_numpy()
        milp_vals = dfv[MILP_COL].to_numpy()
        diffs = cp_vals - milp_vals

        # Basic descriptive
        lines.append(f"  mean(diff = CP - MILP)   = {diffs.mean():.6f}")
        lines.append(f"  median(diff)             = {pd.Series(diffs).median():.6f}")

        # Shapiro
        sh_stat, sh_p, sh_err = safe_shapiro(diffs)
        if sh_err is None:
            lines.append("  Shapiro-Wilk test on differences:")
            lines.append(f"    statistic = {sh_stat:.6f}")
            lines.append(f"    p-value   = {sh_p:.6e}")
            if sh_p < ALPHA:
                lines.append(f"    Interpretation: Reject normality (p < {ALPHA}) -> differences not normal")
            else:
                lines.append(f"    Interpretation: Fail to reject normality (p >= {ALPHA}) -> differences plausibly normal")
        else:
            lines.append("  Shapiro-Wilk test on differences could not be performed: " + str(sh_err))

        # Wilcoxon
        w_stat, w_p, w_err = safe_wilcoxon(cp_vals, milp_vals, alt)
        lines.append("")
        lines.append(f"  Wilcoxon signed-rank test (alternative='{alt}'):")
        if w_err is None:
            lines.append(f"    statistic = {w_stat:.6f}")
            lines.append(f"    p-value   = {w_p:.6e}")
            lines.append(f"    Interpretation: {interpret_wilcoxon_p(w_p, alt)}")
        else:
            lines.append("    Test could not be performed: " + str(w_err))

        lines.append("")

    return "\n".join(lines)


def main():
    try:
        df = load_and_prepare(TABLE_CSV)
    except Exception as e:
        print(f"Error loading or preparing data: {e}", file=sys.stderr)
        sys.exit(2)

    report = run_variant_tests(df, VARIANTS_TO_TEST)

    # Print and save
    print(report)
    OUT_FILE.write_text(report, encoding='utf-8')
    print(f"\nReport saved to: {OUT_FILE}")


if __name__ == '__main__':
    main()
