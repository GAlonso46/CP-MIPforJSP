"""
Utilities to analyze experiment results produced by `experiment/run_experiment.py`.

Provides functions to:
- load raw per-instance JSON result files from `experiment/results/`;
- aggregate repetition-level data into instance-level statistics (median, std, most frequent status);
- produce summary and robustness tables as CSVs under `experiment/tables/`;
- produce plots (boxplot for times, stacked-status breakdown and makespan scatter) under `experiment/graphics/`.

All functions handle missing data gracefully.
"""

from pathlib import Path
import json
import re
from collections import Counter, defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Professional seaborn theme
sns.set_theme(style="whitegrid")


def _detect_size_from_name(name: str) -> str:
    """Detect instance size from its name using a regex for patterns like '5x3' or '10x5'.

    Heuristic:
      - If pattern XxY found, consider 'small' when max(X,Y) <= 5, otherwise 'medium'.
      - If not found, fallback to 'medium'.

    Keeps logic simple and deterministic.
    """
    m = re.search(r"(\d+)x(\d+)", name)
    if not m:
        return "medium"
    try:
        a = int(m.group(1))
        b = int(m.group(2))
    except ValueError:
        return "medium"

    if max(a, b) <= 5:
        return "small"
    return "medium"


def _most_frequent_status(reps):
    if not reps:
        return None
    statuses = [r.get("status") for r in reps if r.get("status") is not None]
    if not statuses:
        return None
    return Counter(statuses).most_common(1)[0][0]


def _extract_rep_values(reps, key: str):
    vals = []
    for r in reps:
        # support both 'obj' and 'obj_val' for objective
        if key == 'obj':
            v = r.get('obj') if 'obj' in r else r.get('obj_val')
        else:
            v = r.get(key)
        vals.append(v)
    return vals


def load_all_results(results_root: Path) -> pd.DataFrame:
    """Walk `results_root` and load all per-instance JSON files into a flat DataFrame.

    The returned DataFrame has one row per instance per solver and includes repetition-level
    aggregated statistics as well as raw repetitions in lists.

    Columns include:
      - variant, instance_name, file_path, size
      - solver ('cp' or 'milp')
      - rep_times (list), rep_objs (list), rep_statuses (list)
      - median_time, std_time, median_obj, std_obj, most_freq_status
    """
    records = []
    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    for variant_dir in sorted(results_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        for file in sorted(variant_dir.iterdir()):
            if not file.is_file() or not file.suffix == '.json':
                continue
            try:
                with file.open('r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                # skip malformed files
                continue

            instance_name = Path(data.get('instance', str(file))).name
            # sometimes instance field contains path; take stem if trailing path
            instance_stem = Path(instance_name).stem
            size = _detect_size_from_name(instance_stem)

            for solver in ('cp', 'milp'):
                solver_obj = data.get(solver)
                if not solver_obj:
                    # produce empty record
                    records.append({
                        'variant': variant_dir.name,
                        'instance_name': instance_stem,
                        'file_path': str(file),
                        'size': size,
                        'solver': solver,
                        'rep_times': [],
                        'rep_objs': [],
                        'rep_statuses': [],
                        'median_time': np.nan,
                        'std_time': np.nan,
                        'median_obj': np.nan,
                        'std_obj': np.nan,
                        'most_freq_status': None,
                    })
                    continue

                reps = solver_obj.get('reps', [])
                times = [r.get('time') for r in reps if r.get('time') is not None]
                objs = [r.get('obj') if 'obj' in r else r.get('obj_val') for r in reps]
                statuses = [r.get('status') for r in reps]

                # convert times/objs to numeric arrays, keeping None as NaN
                times_arr = pd.to_numeric(pd.Series(times), errors='coerce') if times else pd.Series([], dtype=float)
                objs_series = pd.to_numeric(pd.Series(objs), errors='coerce') if objs else pd.Series([], dtype=float)

                median_time = float(times_arr.median()) if not times_arr.empty else np.nan
                std_time = float(times_arr.std(ddof=0)) if not times_arr.empty else np.nan
                median_obj = float(objs_series.median()) if not objs_series.empty and not objs_series.dropna().empty else np.nan
                std_obj = float(objs_series.std(ddof=0)) if not objs_series.empty and not objs_series.dropna().empty else np.nan
                most_freq = _most_frequent_status(reps)

                records.append({
                    'variant': variant_dir.name,
                    'instance_name': instance_stem,
                    'file_path': str(file),
                    'size': size,
                    'solver': solver,
                    'rep_times': times,
                    'rep_objs': objs,
                    'rep_statuses': statuses,
                    'median_time': median_time,
                    'std_time': std_time,
                    'median_obj': median_obj,
                    'std_obj': std_obj,
                    'most_freq_status': most_freq,
                })

    df = pd.DataFrame.from_records(records)
    return df


def aggregate_tables(df: pd.DataFrame, tables_dir: Path) -> Tuple[Path, Path]:
    """Generate summary_table.csv and robustness_table.csv in `tables_dir`.

    Returns paths to the two CSV files.
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Ensure necessary columns exist
    required = {'variant', 'size', 'solver', 'median_time', 'median_obj', 'most_freq_status', 'std_time', 'std_obj'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # SUMMARY TABLE grouped by Variant and Size
    # For both solvers compute average of median_time and median_obj and % OPTIMAL
    def compute_summary(group):
        rows = {}
        total_instances = group['instance_name'].nunique()
        for solver in ('cp', 'milp'):
            g = group[group['solver'] == solver]
            avg_median_time = g['median_time'].replace([np.inf, -np.inf], np.nan).mean()
            avg_median_obj = g['median_obj'].replace([np.inf, -np.inf], np.nan).mean()
            pct_optimal = (g['most_freq_status'] == 'OPTIMAL').sum() / max(1, g['instance_name'].nunique()) * 100
            rows[f'{solver}_avg_median_time'] = avg_median_time
            rows[f'{solver}_avg_median_obj'] = avg_median_obj
            rows[f'{solver}_pct_optimal'] = pct_optimal
        return pd.Series(rows)

    summary = df.groupby(['variant', 'size']).apply(compute_summary).reset_index()
    summary_file = tables_dir / 'summary_table.csv'
    summary.to_csv(summary_file, index=False)

    # ROBUSTNESS TABLE grouped by Variant
    # For each instance compute CoV for time and obj per solver, then average per variant
    cov_records = []
    for (variant, solver), group in df.groupby(['variant', 'solver']):
        # compute per-instance CoV: std_time / mean_time, using rep lists
        covs_time = []
        covs_obj = []
        for _, row in group.iterrows():
            times = pd.to_numeric(pd.Series(row['rep_times']), errors='coerce')
            objs = pd.to_numeric(pd.Series(row['rep_objs']), errors='coerce')
            if not times.dropna().empty and times.dropna().mean() != 0:
                covs_time.append(times.dropna().std(ddof=0) / times.dropna().mean())
            if not objs.dropna().empty and objs.dropna().mean() != 0:
                covs_obj.append(objs.dropna().std(ddof=0) / objs.dropna().mean())

        cov_records.append({
            'variant': variant,
            'solver': solver,
            'avg_cov_time': float(np.nanmean(covs_time)) if covs_time else np.nan,
            'avg_cov_obj': float(np.nanmean(covs_obj)) if covs_obj else np.nan,
        })

    cov_df = pd.DataFrame.from_records(cov_records)
    # pivot so rows=variant and columns for cp/milp
    robustness = cov_df.pivot(index='variant', columns='solver', values=['avg_cov_time', 'avg_cov_obj'])
    # flatten multiindex columns
    robustness.columns = [f"{stat}_{solver}" for stat, solver in robustness.columns]
    robustness = robustness.reset_index()
    robustness_file = tables_dir / 'robustness_table.csv'
    robustness.to_csv(robustness_file, index=False)

    return summary_file, robustness_file


def generate_graphics(df: pd.DataFrame, graphics_dir: Path) -> Tuple[Path, Path, Path]:
    """Generate PNG graphics in `graphics_dir` and return their paths.

    Produces:
      - boxplot_time_variant.png
      - status_breakdown.png
      - makespan_scatter.png
    """
    graphics_dir = Path(graphics_dir)
    graphics_dir.mkdir(parents=True, exist_ok=True)

    # Prepare a long table of repetition-level times with variant and solver
    times_records = []
    for _, row in df.iterrows():
        for t in row['rep_times']:
            if t is None:
                continue
            times_records.append({'variant': row['variant'], 'solver': row['solver'], 'time': float(t)})
    times_df = pd.DataFrame.from_records(times_records)

    # BOXPLOT: computation times CP vs MILP across variants
    plt.figure(figsize=(10, 6))
    if not times_df.empty:
        ax = sns.boxplot(x='variant', y='time', hue='solver', data=times_df, showfliers=False)
        ax.set_yscale('log')
        ax.set_ylabel('Computation time (s) [log scale]')
        ax.set_title('Computation Time by Variant and Solver')
        plt.legend(title='Solver')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'No timing data available', ha='center')
    boxplot_file = graphics_dir / 'boxplot_time_variant.png'
    plt.savefig(boxplot_file, dpi=150)
    plt.close()

    # STATUS BREAKDOWN: stacked bar chart of statuses per variant comparing solvers
    # Use instance-level most frequent status
    status_df = df.groupby(['variant', 'solver', 'most_freq_status']).size().reset_index(name='count')
    # compute percentages per variant and solver
    status_totals = status_df.groupby(['variant', 'solver'])['count'].transform('sum')
    status_df['pct'] = status_df['count'] / status_totals * 100

    # pivot for plotting: for each variant and solver we want a stacked bar of statuses
    pivot = status_df.pivot_table(index=['variant', 'solver'], columns='most_freq_status', values='pct', fill_value=0)
    pivot = pivot.reset_index()

    # create side-by-side grouped bars: one group per variant, two bars (cp,milp)
    variants = pivot['variant'].unique()
    statuses = [c for c in pivot.columns if c not in ('variant', 'solver')]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, solver in enumerate(['cp', 'milp']):
        sub = pivot[pivot['solver'] == solver].set_index('variant').reindex(variants)
        bottoms = np.zeros(len(variants))
        for status in statuses:
            vals = sub[status].values if status in sub.columns else np.zeros(len(variants))
            ax.bar(x + (i - 0.5) * width, vals, width, bottom=bottoms, label=f"{solver.upper()} {status}")
            bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.set_ylabel('Percentage of instances (%)')
    ax.set_title('Status Breakdown by Variant and Solver')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    status_file = graphics_dir / 'status_breakdown.png'
    plt.savefig(status_file, dpi=150)
    plt.close()

    # MAKESPAN SCATTER: median objective CP (X) vs MILP (Y) for instances that have FEASIBLE as most frequent status
    # Build a table with one row per instance containing cp and milp median_obj and most_freq_status
    inst_pivot = df.pivot_table(index=['variant', 'instance_name', 'size', 'file_path'], columns='solver', values=['median_obj', 'most_freq_status'])
    # flatten columns
    inst_pivot.columns = [f'{col[1]}_{col[0]}' for col in inst_pivot.columns]
    inst_pivot = inst_pivot.reset_index()

    # select instances where either solver reached FEASIBLE or TIME_LIMIT (treat TIME_LIMIT as FEASIBLE if desired)
    def is_time_limited(s):
        return s in ("FEASIBLE", "TIME_LIMIT")

    mask = inst_pivot['cp_median_obj'].notna() & inst_pivot['milp_median_obj'].notna() & (
        inst_pivot['cp_most_freq_status'].apply(lambda x: x in ("FEASIBLE", "TIME_LIMIT")) |
        inst_pivot['milp_most_freq_status'].apply(lambda x: x in ("FEASIBLE", "TIME_LIMIT"))
    )

    scatter_df = inst_pivot[mask]

    plt.figure(figsize=(7, 7))
    if not scatter_df.empty:
        sns.scatterplot(x='cp_median_obj', y='milp_median_obj', hue='variant', style='size', data=scatter_df, s=60)
        maxval = max(scatter_df['cp_median_obj'].max(), scatter_df['milp_median_obj'].max())
        minval = min(scatter_df['cp_median_obj'].min(), scatter_df['milp_median_obj'].min())
        # diagonal
        plt.plot([minval, maxval], [minval, maxval], linestyle='--', color='grey')
        plt.xlabel('CP median objective')
        plt.ylabel('MILP median objective')
        plt.title('Median Objective: CP vs MILP (time-limited instances)')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'No instances with FEASIBLE time-limit and valid objectives found', ha='center')

    makespan_file = graphics_dir / 'makespan_scatter.png'
    plt.savefig(makespan_file, dpi=150)
    plt.close()

    return boxplot_file, status_file, makespan_file


def run_full_analysis(results_root: Path, tables_dir: Path, graphics_dir: Path):
    """Convenience function to run the full pipeline: load, aggregate tables, and generate plots.

    Returns tuple of paths: (summary_csv, robustness_csv, boxplot_png, status_png, makespan_png)
    """
    print(f"Loading results from {results_root}")
    df = load_all_results(Path(results_root))
    print(f"Loaded {len(df)} solver-instance records")

    print("Generating tables...")
    summary_file, robustness_file = aggregate_tables(df, Path(tables_dir))
    print(f"Wrote tables to {summary_file} and {robustness_file}")

    print("Generating graphics...")
    boxplot_file, status_file, makespan_file = generate_graphics(df, Path(graphics_dir))
    print(f"Wrote graphics to {boxplot_file}, {status_file}, {makespan_file}")

    return summary_file, robustness_file, boxplot_file, status_file, makespan_file
