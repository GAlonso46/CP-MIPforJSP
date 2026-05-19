#!/usr/bin/env python3
"""
experiment/hipotesis_tests/global_hypothesis_tests.py

Usage:
    python experiment/hipotesis_tests/global_hipotesis_test.py

Reads 'experiment/tables/instances_solved_table.csv', aggregates all paired
cp_median_time and milp_median_time values, runs Shapiro-Wilk on differences
and Wilcoxon signed-rank on pairs, prints results and saves a summary to
experiment/hipotesis_tests/global_test_results.txt.
"""
from pathlib import Path
import os
import sys
import textwrap

import pandas as pd
from scipy import stats

# Paths
ROOT = Path.cwd()
TABLE_CSV = ROOT / "experiment" / "tables" / "instances_solved_table.csv"
OUT_DIR = ROOT / "experiment" / "hipotesis_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "global_test_results.txt"

# Parameters
ALPHA = 0.05
CP_COL = "cp_median_time"
MILP_COL = "milp_median_time"

def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if CP_COL not in df.columns or MILP_COL not in df.columns:
        raise KeyError(f"CSV must contain columns '{CP_COL}' and '{MILP_COL}'")
    # Ensure numeric
    df[CP_COL] = pd.to_numeric(df[CP_COL], errors="coerce")
    df[MILP_COL] = pd.to_numeric(df[MILP_COL], errors="coerce")
    df = df.dropna(subset=[CP_COL, MILP_COL]).reset_index(drop=True)

    # FILTER: Keep rows where NOT (both are >= 600)
    df = df[~((df[CP_COL] >= 600) & (df[MILP_COL] >= 600))].reset_index(drop=True)
    
    return df

def run_tests(cp_vals, milp_vals):
    # paired differences
    diffs = cp_vals - milp_vals

    results = {}

    # Shapiro-Wilk on differences (normality test)
    shapiro_stat = None
    shapiro_p = None
    try:
        shapiro_res = stats.shapiro(diffs)
        shapiro_stat, shapiro_p = float(shapiro_res.statistic), float(shapiro_res.pvalue)
    except Exception as e:
        shapiro_stat, shapiro_p = None, None
        results['shapiro_error'] = str(e)

    results['shapiro_stat'] = shapiro_stat
    results['shapiro_p'] = shapiro_p
    results['shapiro_reject'] = (shapiro_p is not None) and (shapiro_p < ALPHA)

    # Wilcoxon signed-rank test on paired samples
    wilcoxon_stat = None
    wilcoxon_p = None
    try:
        # use two-sided test; zero_method and alternative chosen for robustness
        wil_res = stats.wilcoxon(cp_vals, milp_vals, alternative="two-sided", zero_method="wilcox")
        wilcoxon_stat, wilcoxon_p = float(wil_res.statistic), float(wil_res.pvalue)
    except Exception as e:
        results['wilcoxon_error'] = str(e)

    results['wilcoxon_stat'] = wilcoxon_stat
    results['wilcoxon_p'] = wilcoxon_p
    results['wilcoxon_reject'] = (wilcoxon_p is not None) and (wilcoxon_p < ALPHA)

    # Basic summary stats
    results['n_pairs'] = len(cp_vals)
    results['diff_mean'] = float(diffs.mean()) if len(diffs) > 0 else None
    results['diff_median'] = float(pd.Series(diffs).median()) if len(diffs) > 0 else None

    return results

def format_results(results):
    lines = []
    lines.append("Global paired hypothesis tests (CP vs MILP)")
    lines.append("="*50)
    lines.append(f"Number of paired instances: {results.get('n_pairs')}")
    lines.append("")
    lines.append("Differences (CP - MILP):")
    lines.append(f"  mean = {results.get('diff_mean'):.6f}" if results.get('diff_mean') is not None else "  mean = None")
    lines.append(f"  median = {results.get('diff_median'):.6f}" if results.get('diff_median') is not None else "  median = None")
    lines.append("")

    # Shapiro
    lines.append("Shapiro-Wilk test for normality of differences (H0: differences ~ N):")
    if results.get('shapiro_stat') is not None:
        lines.append(f"  statistic = {results['shapiro_stat']:.6f}")
        lines.append(f"  p-value   = {results['shapiro_p']:.6e}")
        lines.append(f"  alpha     = {ALPHA}")
        lines.append("  Interpretation: " + ("Reject H0 (not normal)" if results['shapiro_reject'] else "Fail to reject H0 (plausibly normal)"))
    else:
        lines.append("  Shapiro test could not be performed.")
        if 'shapiro_error' in results:
            lines.append("  Error: " + results['shapiro_error'])
    lines.append("")

    # Wilcoxon
    lines.append("Wilcoxon signed-rank test on paired samples (H0: median difference = 0):")
    if results.get('wilcoxon_stat') is not None:
        lines.append(f"  statistic = {results['wilcoxon_stat']:.6f}")
        lines.append(f"  p-value   = {results['wilcoxon_p']:.6e}")
        lines.append(f"  alpha     = {ALPHA}")
        lines.append("  Interpretation: " + ("Reject H0 (significant difference)" if results['wilcoxon_reject'] else "Fail to reject H0 (no significant difference)"))
    else:
        lines.append("  Wilcoxon test could not be performed.")
        if 'wilcoxon_error' in results:
            lines.append("  Error: " + results['wilcoxon_error'])
    lines.append("")

    lines.append("Notes:")
    lines.append(" - Shapiro-Wilk null hypothesis: differences are normally distributed.")
    lines.append(" - Wilcoxon null hypothesis: median of paired differences is zero (non-parametric paired test).")
    lines.append(" - alpha = {:.3f}".format(ALPHA))
    return "\n".join(lines)

def main():
    try:
        df = load_data(TABLE_CSV)
    except Exception as e:
        print(f"ERROR loading data: {e}", file=sys.stderr)
        sys.exit(2)

    cp_vals = df[CP_COL].to_numpy()
    milp_vals = df[MILP_COL].to_numpy()

    if cp_vals.size == 0:
        print("No paired data found. Exiting.", file=sys.stderr)
        sys.exit(3)

    results = run_tests(cp_vals, milp_vals)
    summary_text = format_results(results)

    # Print to stdout
    print(summary_text)

    # Save to file
    OUT_FILE.write_text(summary_text, encoding="utf-8")
    print(f"\nSummary saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()