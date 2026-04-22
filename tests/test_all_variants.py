"""
Iterate all instance folders under `instances/`, take the 10 smallest files as described
and solve them with CP, MILP (Gurobi), MILP (SCIP), and MILP (HiGHS) models.
Save results to tests/test_results/<folder>.json
"""
import os
import time
import json
from pathlib import Path
from utils.load_ta import parse_ta_file
from utils.load_json import load_extended_jssp_instance
from models.jssp_cp import JSSPCpModel
from models.jssp_milp import JSSPMilpModel
from models.jssp_scip import JSSPScipModel
from models.jssp_highs import JSSPHighsModel  


def list_smallest_instances(folder: Path):
    """Return 10 smallest filenames according to naming conventions described in the repo."""
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    # Heuristic: for basic_jssp filenames start with 'small', else use suffix '001'..'010'
    if folder.name == 'basic_jssp':
        selected = [f for f in files if f.name.startswith(('small01', 'small02', 'small03', 'small04', 'small05', 'small06', 'small07', 'small08', 'small09', 'small10'))][:10]
    else:
        selected = [f for f in files if f.name.endswith(('001.json','002.json','003.json','004.json','005.json','006.json','007.json','008.json','009.json','010.json'))][:10]
    return selected


def solve_and_record(path: Path, results_dir: Path):
    data = {}
    if path.suffix == '.json':
        data = load_extended_jssp_instance(str(path))
    else:
        data = parse_ta_file(str(path))    

    rec = {
        'instance': str(path),
        'cp': None,
        'milp': None,
        'scip': None,
        'highs': None  
    }

    # 1. CP (Constraint Programming)
    cp_model = JSSPCpModel(data)
    t0 = time.time()
    res_cp = cp_model.optimize(time_limit_seconds=30)
    t1 = time.time()
    rec['cp'] = {
        'status': res_cp.get('status'),
        'obj': res_cp.get('obj_val'),
        'time': t1 - t0,
        'solution': res_cp.get('solution', {})
    }

    # 2. MILP (Gurobi) 
    milp_model = JSSPMilpModel(data)
    t0 = time.time()
    res_milp = milp_model.optimize(time_limit=30)
    t1 = time.time()
    rec['milp'] = {
        'status': res_milp.get('status'),
        'obj': res_milp.get('obj_val'),
        'time': t1 - t0,
        'solution': res_milp.get('solution')
    }

    # 3. MILP (SCIP)
    #scip_model = JSSPScipModel(data)
    #t0 = time.time()
    #res_scip = scip_model.optimize(time_limit=30)
    #t1 = time.time()
    #rec['scip'] = {
    #    'status': res_scip.get('status'),
    #    'obj': res_scip.get('obj_val'),
    #    'time': t1 - t0,
    #    'solution': res_scip.get('solution', {})
    #}

    # 4. MILP (HiGHS)
    #highs_model = JSSPHighsModel(data)
    #t0 = time.time()
    #res_highs = highs_model.optimize(time_limit=30)
    #t1 = time.time()
    #rec['highs'] = {
    #    'status': res_highs.get('status'),
    #    'obj': res_highs.get('obj_val'),
    #    'time': t1 - t0,
    #}

    # Write per-instance record to JSON
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
                print(f"Solved {path.name} -> cp:{rec['cp']['status']} milp:{rec['milp']['status']}")
            except Exception as e:
                print(f"Failed {path.name}: {e}")


if __name__ == '__main__':
    test_all_variants()