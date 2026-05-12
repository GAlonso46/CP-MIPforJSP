"""
Generate summary tables and graphics from experiment results.

Usage:
    python experiment/generate_results.py

This script scans `experiment/results/` and writes CSVs to `experiment/tables/` and PNGs to
`experiment/graphics/`.
"""
from pathlib import Path
import argparse
import sys

from utils.analysis_utils import run_full_analysis


def main(results_dir: Path, tables_dir: Path, graphics_dir: Path):
    print(f"Starting results generation")
    try:
        summary_file, robustness_file, boxplot_file, status_file, makespan_file = run_full_analysis(results_dir, tables_dir, graphics_dir)
        print("Generation completed successfully.")
        print(f"Summary table: {summary_file}")
        print(f"Robustness table: {robustness_file}")
        print(f"Boxplot: {boxplot_file}")
        print(f"Status breakdown: {status_file}")
        print(f"Makespan scatter: {makespan_file}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    base = Path(__file__).parent
    results_root = base / 'results'
    tables_root = base / 'tables'
    graphics_root = base / 'graphics'

    parser = argparse.ArgumentParser(description='Generate tables and graphics from experiment results')
    parser.add_argument('--results', type=Path, default=results_root, help='Path to experiment results root')
    parser.add_argument('--tables', type=Path, default=tables_root, help='Output directory for CSV tables')
    parser.add_argument('--graphics', type=Path, default=graphics_root, help='Output directory for PNG graphics')
    args = parser.parse_args()

    main(args.results, args.tables, args.graphics)
