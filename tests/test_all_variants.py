"""
Iterate all instance folders under `instances/`, take the 10 smallest files as described
and solve them with both CP and MILP models. Save results to tests/test_results/<folder>.json
"""
import os
import time
import json
from pathlib import Path
from utils.load_ta import parse_ta_file
from models.jssp_cp import JSSPCpModel
from models.jssp_milp import JSSPMilpModel


def list_smallest_instances(folder: Path):
    """Return 10 smallest filenames according to naming conventions described in the repo."""
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    # Heuristic: for basic_jssp filenames start with 'small', else use suffix '001'..'010'
    if folder.name == 'basic_jssp':
        selected = [f for f in files if f.name.startswith(('small01', 'small02', 'small03', 'small04', 'small05', 'small06', 'small07', 'small08', 'small09', 'small10'))][:10]
    else:
        selected = [f for f in files if f.name.endswith(('001','002','003','004','005','006','007','008','009','010'))][:10]
    return selected


def solve_and_record(path: Path, results_dir: Path):
    data = parse_ta_file(str(path))

    rec = {
        'instance': str(path),
        'cp': None,
        'milp': None
    }

    # CP
    cp_model = JSSPCpModel(data)
    t0 = time.time()
    res_cp = cp_model.optimize(time_limit_seconds=5)
    t1 = time.time()
    rec['cp'] = {
        'status': res_cp.get('status'),
        'obj': res_cp.get('obj_val'),
        'time': t1 - t0,
        'selected_modes': res_cp.get('solution', {}).get('selected_modes', [])
    }

    # MILP
    milp_model = JSSPMilpModel(data)
    t0 = time.time()
    res_milp = milp_model.optimize(time_limit=5)
    t1 = time.time()
    rec['milp'] = {
        'status': res_milp.get('status'),
        'obj': res_milp.get('obj_val'),
        'time': t1 - t0,
    }

    # write per-instance record
    out_file = results_dir / (path.name + '.json')
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2)

    return rec


def test_all_variants():
    base = Path(__file__).parent.parent
    instances_dir = base / 'instances'
    results_root = Path(__file__).parent / 'test_results'
    results_root.mkdir(exist_ok=True)

    for folder in instances_dir.iterdir():
        if not folder.is_dir():
            continue
        selected = list_smallest_instances(folder)
        folder_results = results_root / folder.name
        folder_results.mkdir(exist_ok=True)
        for path in selected:
            try:
                rec = solve_and_record(path, folder_results)
                print(f"Solved {path} -> cp:{rec['cp']['status']} milp:{rec['milp']['status']}")
            except Exception as e:
                print(f"Failed {path}: {e}")


if __name__ == '__main__':
    test_all_variants()
