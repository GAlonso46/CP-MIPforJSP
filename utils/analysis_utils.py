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

            # preserve the original instance path string when available so downstream code can
            # resolve the instance file; fall back to the actual result file path
            instance_path_val = data.get('instance', str(file))

            for solver in ('cp', 'milp'):
                solver_obj = data.get(solver)
                if not solver_obj:
                    # produce empty record
                    records.append({
                        'variant': variant_dir.name,
                        'instance_name': instance_stem,
                        'file_path': str(file),
                        'instance_path': instance_path_val,
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
                    'instance_path': instance_path_val,
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

    # Ensure numeric columns are numeric: some records can contain strings (from malformed JSON
    # or inconsistent typing). Coerce to numeric and replace non-convertible entries with NaN.
    for col in ('median_time', 'std_time', 'median_obj', 'std_obj'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
    # Compute per (variant,size,solver) aggregated metrics, then pivot solvers into columns.
    grouped = df.groupby(['variant', 'size', 'solver']).agg(
        avg_median_time=('median_time', lambda x: pd.to_numeric(x, errors='coerce').replace([np.inf, -np.inf], np.nan).mean()),
        avg_median_obj=('median_obj', lambda x: pd.to_numeric(x, errors='coerce').replace([np.inf, -np.inf], np.nan).mean()),
        n_instances=('instance_name', 'nunique'),
        n_optimal=('most_freq_status', lambda x: (x == 'OPTIMAL').sum()),
    ).reset_index()

    grouped['pct_optimal'] = grouped.apply(lambda r: (r['n_optimal'] / r['n_instances'] * 100) if r['n_instances'] > 0 else np.nan, axis=1)

    # Now pivot metrics so columns are like cp_avg_median_time, milp_avg_median_time, etc.
    time_pivot = grouped.pivot(index=['variant', 'size'], columns='solver', values='avg_median_time').rename(columns=lambda s: f"{s}_avg_median_time")
    obj_pivot = grouped.pivot(index=['variant', 'size'], columns='solver', values='avg_median_obj').rename(columns=lambda s: f"{s}_avg_median_obj")
    pct_pivot = grouped.pivot(index=['variant', 'size'], columns='solver', values='pct_optimal').rename(columns=lambda s: f"{s}_pct_optimal")

    summary = pd.concat([time_pivot, obj_pivot, pct_pivot], axis=1).reset_index()
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
            # be defensive: some time values might be strings or non-numeric; try to coerce
            try:
                time_val = float(t)
            except Exception:
                # skip non-numeric time entries
                continue
            times_records.append({'variant': row['variant'], 'solver': row['solver'], 'time': time_val})
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
    # Pivot numeric and categorical columns separately to avoid pandas trying to aggregate object dtypes with numeric funcs.
    median_obj_pivot = df.pivot_table(
        index=['variant', 'instance_name', 'size', 'file_path'],
        columns='solver',
        values='median_obj',
        aggfunc='first'
    )

    def top_mode(x):
        x = x.dropna()
        if x.empty:
            return np.nan
        modes = x.mode()
        return modes.iloc[0] if not modes.empty else x.iloc[0]

    status_pivot = df.pivot_table(
        index=['variant', 'instance_name', 'size', 'file_path'],
        columns='solver',
        values='most_freq_status',
        aggfunc=top_mode
    )

    # Normalize column names: ensure columns become 'cp_median_obj', 'milp_median_obj',
    # and 'cp_most_freq_status', 'milp_most_freq_status'. Handle both single- and multi-indexed columns.
    if isinstance(median_obj_pivot.columns, pd.MultiIndex):
        median_obj_pivot.columns = [f"{col[1]}_median_obj" for col in median_obj_pivot.columns]
    else:
        median_obj_pivot = median_obj_pivot.rename(columns=lambda s: f"{s}_median_obj")

    if isinstance(status_pivot.columns, pd.MultiIndex):
        status_pivot.columns = [f"{col[1]}_most_freq_status" for col in status_pivot.columns]
    else:
        status_pivot = status_pivot.rename(columns=lambda s: f"{s}_most_freq_status")

    inst_pivot = pd.concat([median_obj_pivot, status_pivot], axis=1).reset_index()

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


def _resolve_instance_file(instance_path: str, instances_root: Path) -> Path:
    """Resolve instance file path. If instance_path exists return it, otherwise search under instances_root."""
    p = Path(instance_path)
    if p.exists():
        return p
    # try relative to instances_root
    cand = instances_root / p.name
    if cand.exists():
        return cand
    # search by stem
    stem = p.stem
    matches = list(instances_root.rglob(stem + '*'))
    return matches[0] if matches else None


def get_instance_dimensions(instance_path: str, instances_root: Path) -> Tuple[int, int, int]:
    """Return (num_jobs, num_machines, num_workers) inferred from the instance file.

    Handles JSON instances (reads fields or job_tasks/machines) and TA-like plain text files
    where the first non-comment line contains 'jobs machines'.
    """
    inst_file = _resolve_instance_file(instance_path, instances_root)
    if not inst_file:
        return (None, None, None)
    try:
        if inst_file.suffix == '.json':
            with inst_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            num_jobs = data.get('num_jobs') if data.get('num_jobs') is not None else (len(data.get('job_tasks', {})) if isinstance(data.get('job_tasks', {}), dict) else None)
            num_machines = data.get('num_machines') if data.get('num_machines') is not None else (len(data.get('machines', [])) if isinstance(data.get('machines', []), list) else None)
            num_workers = data.get('num_workers') if data.get('num_workers') is not None else (len(data.get('workers', [])) if isinstance(data.get('workers', []), list) else 0)
            return (num_jobs, num_machines, num_workers)
        else:
            with inst_file.open('r', encoding='utf-8') as f:
                for ln in f:
                    line = ln.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = re.split(r"\s+", line)
                    if len(parts) >= 2:
                        try:
                            j = int(parts[0])
                            m = int(parts[1])
                            return (j, m, 0)
                        except Exception:
                            break
            return (None, None, None)
    except Exception:
        return (None, None, None)


def build_master_dataframe(results_root: Path, instances_root: Path) -> pd.DataFrame:
    """Build master DataFrame including instance dimensions and per-solver repetition lists.

    Returns DataFrame similar to load_all_results but with columns J, M, num_workers, Total_Tasks and
    per-record summary statistics (time mean/median/std, obj mean/median/std, counts).
    """
    df = load_all_results(results_root)

    # add dimension columns (coerce None -> np.nan so grouping/means behave)
    Js = []
    Ms = []
    Ws = []
    for ip in df.get('instance_path', pd.Series(dtype=str)).fillna(''):
        j, m, w = get_instance_dimensions(ip, instances_root)
        Js.append(np.nan if j is None else j)
        Ms.append(np.nan if m is None else m)
        Ws.append(np.nan if w is None else w)
    df['J'] = Js
    df['M'] = Ms
    df['num_workers'] = Ws

    # ensure numeric dtypes
    df['J'] = pd.to_numeric(df['J'], errors='coerce')
    df['M'] = pd.to_numeric(df['M'], errors='coerce')
    df['num_workers'] = pd.to_numeric(df['num_workers'], errors='coerce')

    # compute Total_Tasks as numeric (NaN when missing)
    # vectorized multiplication for performance and correctness
    df['Total_Tasks'] = df['J'] * df['M']

    # Per-record aggregation from rep lists
    metrics = []
    for _, row in df.iterrows():
        times = pd.to_numeric(pd.Series(row.get('rep_times', [])), errors='coerce')
        objs = pd.to_numeric(pd.Series(row.get('rep_objs', [])), errors='coerce')
        statuses = pd.Series(row.get('rep_statuses', []))
        count_opt = int((statuses == 'OPTIMAL').sum()) if not statuses.empty else 0
        count_feas = int((statuses == 'FEASIBLE').sum()) if not statuses.empty else 0
        reps_count = int(len(row.get('rep_times', []))) if row.get('rep_times') else 0

        time_mean = float(times.mean()) if not times.dropna().empty else np.nan
        time_median = float(times.median()) if not times.dropna().empty else np.nan
        time_std = float(times.std(ddof=0)) if not times.dropna().empty else np.nan
        obj_mean = float(objs.mean()) if not objs.dropna().empty else np.nan
        obj_median = float(objs.median()) if not objs.dropna().empty else np.nan
        obj_std = float(objs.std(ddof=0)) if not objs.dropna().empty else np.nan
        optimality_rate = (count_opt / reps_count * 100) if reps_count > 0 else np.nan

        metrics.append({
            'time_mean': time_mean,
            'time_median': time_median,
            'time_std': time_std,
            'obj_mean': obj_mean,
            'obj_median': obj_median,
            'obj_std': obj_std,
            'count_optimal': count_opt,
            'count_feasible': count_feas,
            'reps_count': reps_count,
            'optimality_rate': optimality_rate,
        })

    met_df = pd.DataFrame.from_records(metrics)

    # coerce numeric columns to numeric dtype to avoid object dtypes
    for c in ['time_mean', 'time_median', 'time_std', 'obj_mean', 'obj_median', 'obj_std', 'optimality_rate']:
        if c in met_df.columns:
            met_df[c] = pd.to_numeric(met_df[c], errors='coerce')

    df = pd.concat([df.reset_index(drop=True), met_df], axis=1)

    # Final defensive coercion: ensure all numeric-like columns are numeric to avoid pandas errors later
    numeric_like = [c for c in df.columns if any(k in c.lower() for k in ('time', 'obj', 'total', 'j', 'm', 'num_workers', 'optimality_rate'))]
    for c in numeric_like:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            # leave as-is if coercion fails for unexpected reason
            pass

    return df


def generate_variant_deep_dive(df: pd.DataFrame, tables_dir: Path):
    """Create per-variant deep-dive CSV tables grouped by instance dimensions with custom formatting."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    for variant, group in df.groupby('variant'):
        group = group.copy()
        
        # Ensure instance_name is string to avoid NaN in counting
        if 'instance_name' in group.columns:
            group['instance_name'] = group['instance_name'].astype(str)

        # Coerce numeric columns
        num_cols = ['time_mean', 'time_median', 'time_std', 'obj_mean', 'obj_median', 'obj_std', 'Total_Tasks', 'optimality_rate']
        for c in num_cols:
            if c in group.columns:
                group[c] = pd.to_numeric(group[c], errors='coerce')

        group['dim'] = group.apply(lambda r: f"{int(r['J'])}x{int(r['M'])}" if pd.notna(r['J']) and pd.notna(r['M']) else 'unknown', axis=1)
        
        cols = ['dim', 'instance_name', 'solver', 'time_mean', 'time_median', 'time_std', 'obj_mean', 'obj_median', 'obj_std', 'count_optimal', 'count_feasible', 'optimality_rate']
        sub = group[[c for c in cols if c in group.columns]]

        # Aggregation with the specific names requested
        agg = sub.groupby(['dim', 'solver']).agg(
            instances_count=('instance_name', 'nunique'),
            mean_t_mean=('time_mean', 'mean'),
            std_t=('time_median', 'std'),
            mean_obj_mean=('obj_mean', 'mean'),
            std_obj=('obj_median', 'std'),
            std_time_mean=('time_std', 'mean'),
            std_obj_mean=('obj_std', 'mean'),
            opt=('count_optimal', 'sum'),
            feasible=('count_feasible', 'sum'),
            avg_opt=('optimality_rate', 'mean')
        ).reset_index()

        # Pivot the data
        pivot = agg.pivot(index='dim', columns='solver')
        
        # Flatten columns: e.g., (mean_t_mean, cp) -> cp_mean_t_mean
        pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
        
        # Create the single 'count' column (taking it from cp or milp since they are identical)
        if 'cp_instances_count' in pivot.columns:
            pivot.rename(columns={'cp_instances_count': 'count'}, inplace=True)
            if 'milp_instances_count' in pivot.columns:
                pivot.drop(columns=['milp_instances_count'], inplace=True)
        
        pivot = pivot.reset_index()

        # Define the exact order of columns as per your example
        desired_order = [
            'dim', 'count',
            'cp_mean_t_mean', 'milp_mean_t_mean',
            'cp_std_t', 'milp_std_t',
            'cp_mean_obj_mean', 'milp_mean_obj_mean',
            'cp_std_obj', 'milp_std_obj',
            'cp_std_t_mean', 'milp_std_t_mean',
            'cp_std_obj_mean', 'milp_std_obj_mean',
            'cp_opt', 'milp_opt',
            'cp_feas', 'milp_feas',
            'cp_avg_opt', 'milp_avg_opt'
        ]

        # Filter order to include only existing columns to avoid errors
        final_cols = [c for c in desired_order if c in pivot.columns]
        pivot = pivot[final_cols]

        out_file = tables_dir / f"variant_deep_dive_{variant}.csv"
        pivot.to_csv(out_file, index=False)


def generate_global_scalability(df: pd.DataFrame, tables_dir: Path):
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # coerce numeric columns used in aggregations
    for c in ['time_median', 'time_mean', 'obj_median', 'obj_mean', 'optimality_rate']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['dim'] = df.apply(lambda r: f"{int(r['J'])}x{int(r['M'])}" if pd.notna(r['J']) and pd.notna(r['M']) else 'unknown', axis=1)

    try:
        # replace medians with stddevs per requirement
        agg = df.groupby(['dim', 'solver']).agg(
            std_time=('time_median', lambda x: pd.to_numeric(x, errors='coerce').std()),
            mean_time=('time_mean', lambda x: pd.to_numeric(x, errors='coerce').mean()),
            std_obj=('obj_median', lambda x: pd.to_numeric(x, errors='coerce').std()),
            mean_obj=('obj_mean', lambda x: pd.to_numeric(x, errors='coerce').mean()),
            avg_opt_rate=('optimality_rate', lambda x: pd.to_numeric(x, errors='coerce').mean())
        ).reset_index()
    except Exception:
        # coerce and retry
        for c in ['time_median', 'time_mean', 'obj_median', 'obj_mean', 'optimality_rate']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        agg = df.groupby(['dim', 'solver']).agg(
            std_time=('time_median', 'std'),
            mean_time=('time_mean', 'mean'),
            std_obj=('obj_median', 'std'),
            mean_obj=('obj_mean', 'mean'),
            avg_opt_rate=('optimality_rate', 'mean')
        ).reset_index()

    out = agg.pivot(index='dim', columns='solver')
    out.columns = [f"{col[1]}_{col[0]}" for col in out.columns]
    out = out.reset_index()

    # coerce all numeric output columns
    for c in out.columns:
        if c != 'dim':
            out[c] = pd.to_numeric(out[c], errors='coerce')

    out.to_csv(Path(tables_dir) / 'global_scalability.csv', index=False)


def generate_additional_plots(df: pd.DataFrame, graphics_dir: Path):
    graphics_dir = Path(graphics_dir)
    graphics_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Copy and type cleaning for numeric columns
    df = df.copy()
    numeric_cols = ['Total_Tasks', 'time_median', 'time_mean', 'obj_median', 'obj_mean', 'optimality_rate', 'obj_std']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 2. Scaling Plot
    # Group by Total_Tasks and Solver using the mean of time_mean
    ag = df.groupby(['Total_Tasks', 'solver'])['time_mean'].mean().reset_index()
    ag = ag.dropna(subset=['Total_Tasks'])
    
    plt.figure(figsize=(8, 5))
    if not ag.empty:
        sns.lineplot(data=ag, x='Total_Tasks', y='time_mean', hue='solver', marker='o')
        plt.xlabel('Total Tasks (J*M)')
        plt.ylabel('Mean Time (s)')
        plt.title('Scaling: Mean Resolution Time vs Problem Size')
        plt.tight_layout()
        plt.savefig(graphics_dir / 'scaling_plot_time.png', dpi=150)
    plt.close()

    # 3. Optimality Heatmap
    # Ensure 'dim' exists as string to avoid type errors
    df['dim'] = df.apply(lambda r: f"{int(r['J'])}x{int(r['M'])}" if pd.notna(r['J']) and pd.notna(r['M']) else 'unknown', axis=1)
    
    opt = df.groupby(['variant', 'dim', 'solver'])['optimality_rate'].mean().reset_index()
    
    # Safe pivot for the Heatmap
    opt_pivot = opt.pivot_table(index='dim', columns=['variant', 'solver'], values='optimality_rate')
    
    # Extract CP and MILP separately for subtraction
    cp_vals = opt_pivot.xs('cp', level='solver', axis=1)
    milp_vals = opt_pivot.xs('milp', level='solver', axis=1)
    
    # Robust subtraction (CP - MILP): Positive means CP is better at finding optima
    diff = cp_vals.sub(milp_vals, fill_value=0)
    
    plt.figure(figsize=(12, max(4, diff.shape[0] * 0.4)))
    sns.heatmap(diff, annot=True, fmt='.1f', cmap='coolwarm', center=0)
    plt.title('Optimality Rate Delta (CP % - MILP %)')
    plt.tight_layout()
    plt.savefig(graphics_dir / 'optimality_heatmap.png', dpi=150)
    plt.close()

    # 4. Relative Gap Analysis (Boxplot)
    # Pivot table with explicit aggfunc to avoid errors with 'most_freq_status' (NoneType/isin)
    inst = df.pivot_table(
        index=['variant', 'instance_name', 'dim'], 
        columns='solver', 
        values=['obj_mean', 'most_freq_status'],
        aggfunc={'obj_mean': 'mean', 'most_freq_status': 'first'}
    )
    
    # Flatten columns: ['cp_obj_mean', 'milp_obj_mean', 'cp_most_freq_status', ...]
    inst.columns = [f"{col[1]}_{col[0]}" for col in inst.columns]
    inst = inst.reset_index()

    # Safety check to prevent 'AttributeError' in .isin()
    status_cols = ['cp_most_freq_status', 'milp_most_freq_status']
    if all(col in inst.columns for col in status_cols):
        # Convert to string and fill NAs so .isin() does not fail if Nones exist
        for col in status_cols:
            inst[col] = inst[col].astype(str).replace('nan', 'UNKNOWN')

        mask = (
            (inst['cp_most_freq_status'].isin(['FEASIBLE', 'TIME_LIMIT'])) |
            (inst['milp_most_freq_status'].isin(['FEASIBLE', 'TIME_LIMIT']))
        ) & inst['cp_obj_mean'].notna() & inst['milp_obj_mean'].notna()
        
        inst_sub = inst[mask].copy()
        
        if not inst_sub.empty:
            # Formula: (MILP - CP) / CP * 100. Positive = CP better (lower makespan)
            inst_sub['rel_gap'] = ((inst_sub['milp_obj_mean'] - inst_sub['cp_obj_mean']) / inst_sub['cp_obj_mean']) * 100
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='variant', y='rel_gap', data=inst_sub)
            plt.axhline(0, linestyle='--', color='black', alpha=0.5)
            plt.ylabel('Relative Gap % ((MILP - CP) / CP)')
            plt.title('Solution Quality Gap (Instances with Time Limit)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(graphics_dir / 'gap_analysis_boxplot.png', dpi=150)
    plt.close()


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


def run_deep_analysis(results_root: Path, tables_dir: Path, graphics_dir: Path):
    """High-level entry to run the new deep analysis and produce tables and plots.

    This version defensively coerces all non-categorical columns to numeric (non-convertible -> NaN)
    to avoid pandas raising "dtype 'str' does not support operation 'mean'" during groupby/agg.
    """
    instances_root = Path(__file__).parent.parent / 'instances'
    print(f"Instances root: {instances_root}")
    master = build_master_dataframe(results_root, instances_root)

    # Define categorical columns we must preserve
    categorical_cols = {
        'variant', 'solver', 'instance_name', 'file_path', 'size', 'most_freq_status',
        'rep_times', 'rep_objs', 'rep_statuses', 'instance_path'
    }

    # Coerce everything else to numeric (strings that look numeric will convert; others -> NaN)
    for col in list(master.columns):
        if col in categorical_cols:
            continue
        try:
            master[col] = pd.to_numeric(master[col], errors='coerce')
        except Exception:
            # if conversion fails for unexpected reason, leave original column
            pass

    # Quick diagnostics
    print('Master frame columns and dtypes:')
    try:
        print(master.dtypes)
    except Exception:
        print('Unable to print dtypes')

    for c in ['time_median', 'time_mean', 'obj_median', 'obj_mean', 'Total_Tasks', 'optimality_rate']:
        if c in master.columns:
            n_total = len(master)
            n_nonnull = int(master[c].notna().sum())
            print(f"Column {c}: {n_nonnull}/{n_total} non-null after coercion; dtype={master[c].dtype}")

    # Run generation steps with diagnostics on failure
    try:
        generate_variant_deep_dive(master, tables_dir)
        generate_global_scalability(master, tables_dir)
        generate_additional_plots(master, graphics_dir)
    except Exception as e:
        print('\nERROR during deep analysis:')
        print(repr(e))
        print('\nMaster frame sample:')
        try:
            print(master.head(10))
        except Exception:
            pass
        print('\nMaster dtypes:')
        try:
            print(master.dtypes)
        except Exception:
            pass
        # try to show first non-numeric entries for key columns
        keys = ['time_median', 'time_mean', 'obj_median', 'obj_mean', 'optimality_rate', 'Total_Tasks']
        for k in keys:
            if k in master.columns:
                non_numeric = master[~master[k].apply(lambda v: pd.isna(v) or isinstance(v, (int, float, np.floating, np.integer)))][k]
                if not non_numeric.empty:
                    print(f"First non-numeric values in {k} (showing up to 10):\n", non_numeric.head(10))
        raise
