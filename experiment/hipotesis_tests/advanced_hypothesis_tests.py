#!/usr/bin/env python3
"""
experiment/hipotesis_tests/advanced_hypothesis_tests.py

Performs:
- Phase 1: Medium-size subset paired hypothesis tests (Shapiro-Wilk, Wilcoxon) for selected variants.
- Phase 2: Scalability trend analysis (Spearman correlation between instance complexity and time gap) for selected variants.

Results are saved to experiment/hipotesis_tests/advanced_tests_results.txt
"""

from pathlib import Path
import sys
import pandas as pd
from scipy import stats

# --- Configuration ---
ROOT = Path.cwd()
TABLE_CSV = ROOT / "experiment" / "tables" / "instances_solved_table.csv"
OUT_DIR = ROOT / "experiment" / "hipotesis_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "advanced_tests_results.txt"

CP_COL = "cp_median_time"
MILP_COL = "milp_median_time"
VARIANT_COL = "variant"
SIZE_COL = "dim"
ALPHA = 0.05

# Phase 1: Medium-size subset
MEDIUM_SIZES = ['8x8', '10x5', '15x5', '20x5', '10x10']
PHASE1_VARIANTS = [
    ("drc_jssp", "less"),      # one-tailed: CP faster
    ("sdst_jssp", "less"),     # one-tailed: CP faster
    ("f_jssp", "greater"),     # one-tailed: MILP faster
    ("basic_jssp", "two-sided"),
    ("d_jssp", "two-sided"),
    ("r_jssp", "two-sided"),
    ("t_jssp", "two-sided"),
]

# Phase 2: Scalability trend
PHASE2_VARIANTS = [
    ("drc_jssp", "greater"),      # one-tailed: rho > 0 (MILP increases faster)
    ("sdst_jssp", "greater"),     # one-tailed: rho > 0
    ("f_jssp", "less"),           # one-tailed: rho < 0 (CP increases faster)
    ("basic_jssp", "two-sided"),
    ("d_jssp", "two-sided"),
    ("r_jssp", "two-sided"),
    ("t_jssp", "two-sided"),
]

# --- Utility Functions ---

def load_and_prepare(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    for c in (CP_COL, MILP_COL, VARIANT_COL, SIZE_COL):
        if c not in df.columns:
            raise KeyError(f"Required column missing from CSV: {c}")
    df[CP_COL] = pd.to_numeric(df[CP_COL], errors="coerce")
    df[MILP_COL] = pd.to_numeric(df[MILP_COL], errors="coerce")
    df = df.dropna(subset=[CP_COL, MILP_COL, VARIANT_COL, SIZE_COL]).reset_index(drop=True)
    # Filter out censored ties
    mask_both_censored = (df[CP_COL] >= 600) & (df[MILP_COL] >= 600)
    df = df[~mask_both_censored].copy()
    return df

def safe_shapiro(diffs):
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
        return ("Reject H0: CP is faster (CP < MILP)" if reject
                else "Fail to reject H0: no evidence CP is faster")
    if alt == 'greater':
        return ("Reject H0: MILP is faster (CP > MILP)" if reject
                else "Fail to reject H0: no evidence MILP is faster")
    if reject:
        return "Reject H0: evidence of a difference between CP and MILP"
    return "Fail to reject H0: no evidence of a difference"

def safe_spearman(x, y):
    try:
        res = stats.spearmanr(x, y, nan_policy='omit')
        return float(res.correlation), float(res.pvalue), None
    except Exception as e:
        return None, None, str(e)

def interpret_spearman(rho, p, alt):
    if rho is None or p is None:
        return "Test could not be performed"
    reject = p < ALPHA
    if alt == "greater":
        return ("Reject H0: rho > 0 (MILP time increases faster with size)" if reject
                else "Fail to reject H0: no evidence MILP scales worse")
    if alt == "less":
        return ("Reject H0: rho < 0 (CP time increases faster with size)" if reject
                else "Fail to reject H0: no evidence CP scales worse")
    if reject:
        return "Reject H0: significant monotonic association (rho != 0)"
    return "Fail to reject H0: no significant monotonic association"

def parse_complexity(dim_str):
    try:
        a, b = dim_str.lower().split('x')
        return int(a) * int(b)
    except Exception:
        return None

# --- Phase 1: Medium-Size Subset Hypothesis Testing ---

def phase1_medium_size_tests(df, variants, sizes):
    lines = []
    lines.append("PHASE 1: Medium-Size Subset Hypothesis Testing")
    lines.append("="*70)
    df_med = df[df[SIZE_COL].isin(sizes)].copy()
    lines.append(f"Medium-size dimensions: {sizes}")
    lines.append(f"Total rows after filtering: {len(df_med)}")
    lines.append("")

    for variant, alt in variants:
        lines.append("-"*60)
        lines.append(f"Variant: {variant}")
        dfv = df_med[df_med[VARIANT_COL] == variant]
        n = len(dfv)
        lines.append(f"Sample size: {n}")
        if n == 0:
            lines.append("No data for this variant after filtering.\n")
            continue
        cp_vals = dfv[CP_COL].to_numpy()
        milp_vals = dfv[MILP_COL].to_numpy()
        diffs = cp_vals - milp_vals
        lines.append(f"  mean(diff = CP - MILP):   {diffs.mean():.6f}")
        lines.append(f"  median(diff):            {pd.Series(diffs).median():.6f}")

        # Shapiro-Wilk
        sh_stat, sh_p, sh_err = safe_shapiro(diffs)
        if sh_err is None:
            lines.append("  Shapiro-Wilk test on differences:")
            lines.append(f"    statistic = {sh_stat:.6f}")
            lines.append(f"    p-value   = {sh_p:.6e}")
            if sh_p < ALPHA:
                lines.append(f"    Interpretation: Reject normality (p < {ALPHA})")
            else:
                lines.append(f"    Interpretation: Fail to reject normality (p >= {ALPHA})")
        else:
            lines.append("  Shapiro-Wilk test could not be performed: " + str(sh_err))

        # Wilcoxon
        w_stat, w_p, w_err = safe_wilcoxon(cp_vals, milp_vals, alt)
        lines.append("")
        lines.append(f"  Wilcoxon signed-rank test (alternative='{alt}'):")
        if w_err is None:
            lines.append(f"    statistic = {w_stat:.6f}")
            lines.append(f"    p-value   = {w_p:.6e}")
            lines.append(f"    Conclusion: {interpret_wilcoxon_p(w_p, alt)}")
        else:
            lines.append("    Test could not be performed: " + str(w_err))
        lines.append("")
    return "\n".join(lines)

# --- Phase 2: Scalability Trend Analysis ---

def phase2_scalability_trend(df, variants):
    lines = []
    lines.append("PHASE 2: Scalability Trend Analysis (Spearman Correlation)")
    lines.append("="*70)
    # Add complexity and time_gap columns
    df = df.copy()
    df['complexity'] = df[SIZE_COL].apply(parse_complexity)
    df['time_gap'] = df[MILP_COL] - df[CP_COL]
    lines.append("complexity = jobs * machines (parsed from 'dim' column)")
    lines.append("time_gap = milp_median_time - cp_median_time")
    lines.append("")

    for variant, alt in variants:
        lines.append("-"*60)
        lines.append(f"Variant: {variant}")
        dfv = df[df[VARIANT_COL] == variant]
        n = len(dfv)
        lines.append(f"Sample size: {n}")
        if n == 0:
            lines.append("No data for this variant.\n")
            continue
        x = dfv['complexity']
        y = dfv['time_gap']
        # Remove rows with missing complexity
        mask = x.notnull() & y.notnull()
        x = x[mask]
        y = y[mask]
        n_eff = len(x)
        lines.append(f"  (after removing missing complexity/time_gap: {n_eff})")
        if n_eff < 3:
            lines.append("  Not enough data for correlation.\n")
            continue
        rho, p, err = safe_spearman(x, y)
        if err is None:
            lines.append(f"  Spearman rho = {rho:.6f}")
            lines.append(f"  p-value      = {p:.6e}")
            lines.append(f"  Conclusion: {interpret_spearman(rho, p, alt)}")
        else:
            lines.append("  Spearman correlation could not be performed: " + str(err))
        lines.append("")
    return "\n".join(lines)

# --- Main ---

def main():
    try:
        df = load_and_prepare(TABLE_CSV)
    except Exception as e:
        print(f"Error loading or preparing data: {e}", file=sys.stderr)
        sys.exit(2)

    report = []
    report.append("# Advanced JSSP Hypothesis Testing Results")
    report.append("")
    report.append(phase1_medium_size_tests(df, PHASE1_VARIANTS, MEDIUM_SIZES))
    report.append("\n" + "="*80 + "\n")
    report.append(phase2_scalability_trend(df, PHASE2_VARIANTS))

    report_text = "\n".join(report)
    print(report_text)
    OUT_FILE.write_text(report_text, encoding='utf-8')
    print(f"\nReport saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()