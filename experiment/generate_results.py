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

from utils.analysis_utils import run_full_analysis, run_deep_analysis


def main(results_dir: Path, tables_dir: Path, graphics_dir: Path):
    #print("=" * 60)
    #print("Starting standard results generation")
    #print("=" * 60)
    #try:
    #    summary_file, robustness_file, boxplot_file, status_file, makespan_file = run_full_analysis(
    #        results_dir, tables_dir, graphics_dir
    #    )
    #    print("\nStandard generation completed successfully.")
    #    print(f"Summary table: {summary_file}")
    #    print(f"Robustness table: {robustness_file}")
    #    print(f"Boxplot: {boxplot_file}")
    #    print(f"Status breakdown: {status_file}")
    #    print(f"Makespan scatter: {makespan_file}")
    #except Exception as e:
    #    print(f"Error during standard generation: {e}")
    #    sys.exit(1)

    print("\n" + "=" * 60)
    print("Starting deep analysis generation")
    print("=" * 60)
    try:
        run_deep_analysis(results_dir, tables_dir, graphics_dir)
        # generate instances solved table as part of deep analysis outputs
        try:
            from utils.analysis_utils import generate_instances_solved_table
            instances_table = generate_instances_solved_table(results_dir, tables_dir)
            print(f"Instances solved table written to: {instances_table}")
        except Exception as ee:
            print(f"Could not generate instances solved table: {ee}")
        print("\nDeep analysis completed successfully.")
    except Exception as e:
        print(f"Error during deep analysis: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All generation tasks completed successfully.")
    print("=" * 60)


if __name__ == '__main__':
    base = Path(__file__).parent
    results_root = base / 'results'
    tables_root = base / 'tables'
    graphics_root = base / 'graphics'

    parser = argparse.ArgumentParser(description='Generate tables and graphics from experiment results (standard + deep analysis)')
    parser.add_argument('--results', type=Path, default=results_root, help='Path to experiment results root')
    parser.add_argument('--tables', type=Path, default=tables_root, help='Output directory for CSV tables')
    parser.add_argument('--graphics', type=Path, default=graphics_root, help='Output directory for PNG graphics')
    args = parser.parse_args()

    main(args.results, args.tables, args.graphics)
