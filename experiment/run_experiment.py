"""
Run a set of experiments over the instances in `instances/`.

This script is similar to `tests/test_all_variants.py` but:
- Solves each instance with CP and MILP (Gurobi) models.
- Groups instances by size (small, mid, large) with three helper functions.
- Supports multiple consecutive repetitions per solver and records status/obj/time for
  every repetition while only storing the solution from the first repetition.
- Writes one JSON file per instance inside `experiment/results/<folder>/`.

Usage (optional):
    python experiment/run_experiment.py [--group small|mid|large|all] [--repeats N]

All docstrings and comments are in English.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Optional

from utils.load_ta import parse_ta_file
from utils.load_json import load_extended_jssp_instance
from models.jssp_cp import JSSPCpModel
from models.jssp_milp import JSSPMilpModel


# Default number of repetitions per solver
REPEATS = 3
# Default time limit (seconds) per optimization call
TIME_LIMIT = 30


def _extract_trailing_number(fname: str) -> Optional[int]:
    """Return trailing three-digit number from filename like '...001.json' or None."""
    m = re.search(r"(\d{3})\.json$", fname)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def list_small_instances(folder: Path) -> List[Path]:
    """Return all "small" instances in the folder.

    Small instances are defined as:
    - files whose name starts with 'small' (e.g. 'small01', with or without extension), OR
    - files whose filename ends with a numeric suffix between 001 and 040 (inclusive), e.g. 'd_jssp001.json'.
    """
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    selected: List[Path] = []

    for f in files:
        name = f.name
        # name starts with 'small' (case sensitive, following repo convention)
        if name.startswith('small'):
            selected.append(f)
            continue
        # numeric suffix 001..040
        num = _extract_trailing_number(name)
        if num is not None and 1 <= num <= 40:
            selected.append(f)

    return selected


def list_mid_instances(folder: Path) -> List[Path]:
    """Return all medium-size instances in the folder.

    Medium instances are defined as:
    - files whose name starts with 'la' (e.g. 'la01', with or without extension), OR
    - files whose filename ends with a numeric suffix between 041 and 075 (inclusive).
    """
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    selected: List[Path] = []

    for f in files:
        name = f.name
        if name.startswith('la'):
            selected.append(f)
            continue
        num = _extract_trailing_number(name)
        if num is not None and 41 <= num <= 75:
            selected.append(f)

    return selected


def list_large_instances(folder: Path) -> List[Path]:
    """Return all large instances in the folder.

    Large instances are defined as:
    - files whose name starts with 'ta' (e.g. 'ta01', with or without extension), OR
    - files whose filename ends with a numeric suffix between 076 and 115 (inclusive).
    """
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    selected: List[Path] = []

    for f in files:
        name = f.name
        if name.startswith('ta'):
            selected.append(f)
            continue
        num = _extract_trailing_number(name)
        if num is not None and 76 <= num <= 115:
            selected.append(f)

    return selected


def _load_instance(path: Path):
    """Load an instance file. If JSON use load_extended_jssp_instance, otherwise parse TA file."""
    if path.suffix == '.json':
        return load_extended_jssp_instance(str(path))
    return parse_ta_file(str(path))


def solve_instance_repeatedly(data, model_name: str, repeats: int, time_limit: int):
    """Solve the same instance `repeats` times using the requested model.

    Returns a dict with:
      - 'reps': list of {status, obj, time} for every repetition
      - 'solution': solution object from the first repetition (or None if absent)
    """
    reps = []
    first_solution = None

    for i in range(repeats):
        t0 = time.time()
        if model_name == 'cp':
            model = JSSPCpModel(data)
            res = model.optimize(time_limit_seconds=time_limit)
        elif model_name == 'milp':
            model = JSSPMilpModel(data)
            res = model.optimize(time_limit=time_limit)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        t1 = time.time()

        reps.append({
            'status': res.get('status'),
            'obj': res.get('obj_val'),
            'time': t1 - t0,
        })

        if i == 0:
            # store full solution only from the first repetition
            first_solution = res.get('solution')

    return {'reps': reps, 'solution': first_solution}


def process_folder(folder: Path, selector_func, results_root: Path, repeats: int, time_limit: int):
    """Process one folder of instances using the selector function.

    Writes one JSON per instance in results_root / folder.name.
    """
    selected = selector_func(folder)
    folder_results = results_root / folder.name
    folder_results.mkdir(parents=True, exist_ok=True)

    for path in selected:
        try:
            data = _load_instance(path)

            rec = {
                'instance': str(path),
                'repeats': repeats,
                'cp': None,
                'milp': None,
            }

            # CP repeated runs
            cp_record = solve_instance_repeatedly(data, 'cp', repeats, time_limit)
            rec['cp'] = {
                'reps': cp_record['reps'],
                'solution': cp_record['solution'] or {},
            }

            # MILP (Gurobi) repeated runs
            milp_record = solve_instance_repeatedly(data, 'milp', repeats, time_limit)
            rec['milp'] = {
                'reps': milp_record['reps'],
                'solution': milp_record['solution'] or None,
            }

            # Write per-instance record to JSON (filename preserved as in tests)
            out_file = folder_results / (path.name + '.json')
            with out_file.open('w', encoding='utf-8') as f:
                json.dump(rec, f, indent=2)

            print(f"Solved {path} -> cp:{rec['cp']['reps'][-1]['status']} milp:{rec['milp']['reps'][-1]['status']}")

        except Exception as e:
            print(f"Failed {path}: {e}")


def run(group: str = 'all', repeats: int = REPEATS, time_limit: int = TIME_LIMIT):
    """Run experiments over the instances grouped by `group`.

    group: 'small', 'mid', 'large', or 'all'
    repeats: number of repeated solves per solver
    time_limit: per-call time limit (seconds)
    """
    base = Path(__file__).parent.parent
    instances_dir = base / 'instances'
    results_root = Path(__file__).parent / 'results'
    results_root.mkdir(parents=True, exist_ok=True)

    if group not in ('small', 'mid', 'large', 'all'):
        raise ValueError("group must be one of: small, mid, large, all")

    for folder in instances_dir.iterdir():
        if not folder.is_dir():
            continue

        if group == 'small':
            selector = list_small_instances
        elif group == 'mid':
            selector = list_mid_instances
        elif group == 'large':
            selector = list_large_instances
        else:
            # when 'all', process all sizes in the same run: select everything
            # combine the three selectors to preserve ordering without duplicates
            def selector(f: Path):
                seen = set()
                combined = []
                for s in (list_small_instances(f) + list_mid_instances(f) + list_large_instances(f)):
                    if s not in seen:
                        combined.append(s)
                        seen.add(s)
                return combined

        process_folder(folder, selector, results_root, repeats, time_limit)


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiment over JSSP instances')
    parser.add_argument('--group', choices=['small', 'mid', 'large', 'all'], default='all', help='Which size group to run')
    parser.add_argument('--repeats', type=int, default=REPEATS, help='Number of repetitions per solver')
    parser.add_argument('--time-limit', type=int, default=TIME_LIMIT, help='Time limit (seconds) per optimization call')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(group=args.group, repeats=args.repeats, time_limit=args.time_limit)
