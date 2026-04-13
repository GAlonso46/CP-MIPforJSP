"""
Test that solves a representative TA instance (ta50) with both CP and MILP models.
Prints objective value, assignment and solve time for each solver.
"""
import time
import os
from utils.load_ta import parse_ta_file
from models.jssp_cp import JSSPCpModel
from models.jssp_milp import JSSPMilpModel


def _print_cp_solution(res):
    status = res.get("status")
    obj = res.get("obj_val")
    sol = res.get("solution", {})
    selected = sol.get("selected_modes", [])
    tasks = sol.get("tasks", {})
    print(f"CP result: status={status}, obj={obj}")
    print(" Selected modes:")
    for (t, m, w) in selected:
        print(f"  - Task {t}: machine={m}, worker={w}")
    print(" Task times:")
    for t, info in tasks.items():
        print(f"  - {t}: start={info['start']} end={info['end']}")


def _print_milp_solution(model, res):
    status = res.get("status")
    obj = res.get("obj_val")
    print(f"MILP result: status={status}, obj={obj}")
    # print assignments
    assigns = []
    for (t, m, w), var in model.x.items():
        try:
            val = var.X
        except Exception:
            val = getattr(var, 'X', None)
        if val is not None and val > 0.5:
            assigns.append((t, m, w))
    print(" Assignments:")
    for (t, m, w) in assigns:
        print(f"  - Task {t}: machine={m}, worker={w}")


def test_solve_ta50():
    """Load instances/basic_jssp/ta50 and solve with CP and MILP, printing results."""
    base = os.path.dirname(os.path.dirname(__file__))
    ta_path = os.path.join(base, "instances", "basic_jssp", "ta50")
    print(f"Loading TA instance: {ta_path}")
    data = parse_ta_file(ta_path)

    # Solve with CP
    cp_model = JSSPCpModel(data)
    t0 = time.time()
    res_cp = cp_model.optimize(time_limit_seconds=10)
    t1 = time.time()
    print(f"CP solve time: {t1 - t0:.3f} s")
    _print_cp_solution(res_cp)

    # Solve with MILP
    milp_model = JSSPMilpModel(data)
    t0 = time.time()
    res_milp = milp_model.optimize(time_limit=10)
    t1 = time.time()
    print(f"MILP solve time: {t1 - t0:.3f} s")
    _print_milp_solution(milp_model, res_milp)


if __name__ == '__main__':
    test_solve_ta50()
