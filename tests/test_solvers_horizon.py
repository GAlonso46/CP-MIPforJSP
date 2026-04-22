"""
Test that solves a representative TA instance with both CP and MILP models.
Prints objective value, assignment and solve time for each solver.
"""
import time
import os
from utils.load_ta import parse_ta_file
from models.jssp_cp import JSSPCpModel
from models.jssp_milp import JSSPMilpModel
from models.jssp_scip import JSSPScipModel


def _print_cp_solution(res):
    status = res.get("status")
    obj = res.get("obj_val")
    sol = res.get("solution", {})
    selected = sol.get("selected_modes", [])
    tasks = sol.get("tasks", {})
    print(f"CP result: status={status}, obj={obj}")
    #print(" Selected modes:")
    #for (t, m, w) in selected:
    #    print(f"  - Task {t}: machine={m}, worker={w}")
    #print(" Task times:")
    #for t, info in tasks.items():
    #    print(f"  - {t}: start={info['start']} end={info['end']}")


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
    #print(" Assignments:")
    #for (t, m, w) in assigns:
    #    print(f"  - Task {t}: machine={m}, worker={w}")


def _print_scip_solution(model_obj, res):
    """
    Prints results for the SCIP model.
    Note: SCIP variable values are accessed via model.getVal(var).
    """
    status = res.get("status")
    obj = res.get("obj_val")
    print(f"MILP (SCIP) result: status={status}, obj={obj}")
    
    if obj is None:
        return

    assigns = []
    # Access the PySCIPOpt Model instance from the wrapper
    scip_m = model_obj.model 
    
    for (t, m, w), var in model_obj.x.items():
        try:
            # PySCIPOpt way to get variable value
            val = scip_m.getVal(var)
            if val is not None and val > 0.5:
                assigns.append((t, m, w))
        except Exception:
            continue
            
    #print(" Assignments:")
    #for (t, m, w) in assigns:
    #    print(f"  - Task {t}: machine={m}, worker={w}")        


def test_solve_ta():
    """Load instances/basic_jssp/la10 and solve with CP and MILP, printing results."""
    base = os.path.dirname(os.path.dirname(__file__))
    ta_path = os.path.join(base, "instances", "basic_jssp", "la10")
    print(f"Loading TA instance: {ta_path}")
    data = parse_ta_file(ta_path)

    # Print parsed instance
    #print("Parsed instance data:")
    #print(f" Machines: {data.get('machines')}")
    #print(f" Jobs: {data.get('jobs')}")
    #print(f" Tasks: {data.get('tasks')}")
    #print(f" Precedence arcs: {data.get('P')}")
    #print(f"Processing times (t,m,w): {data.get('p')}")

    params_test_0 = {
        "max_time_in_seconds": 1800,
        "num_search_workers": 2,           
        "log_search_progress": False,       
        "linearization_level": 1,           
        "cp_model_presolve": False,         
    }

    params_test_1 = {
        "max_time_in_seconds": 1800,
        "num_search_workers": 4,           
        "log_search_progress": False,       
        "linearization_level": 1,           
        "cp_model_presolve": False,         
    }

    params_test_2 = {
        "max_time_in_seconds": 600,
        "num_search_workers": 4,           
        "log_search_progress": False,       
        "linearization_level": 2,           
        "cp_model_presolve": False,         
    }

    # Solve with CP
    print("Using parms_test_2")
    cp_model = JSSPCpModel(data, cp_params=params_test_2)
    t0 = time.time()
    res_cp = cp_model.optimize()
    t1 = time.time()
    print(f"CP solve time: {t1 - t0:.3f} s")
    _print_cp_solution(res_cp)

    # Solve with MILP (Gurobi)
    milp_model = JSSPMilpModel(data)
    t0 = time.time()
    res_milp = milp_model.optimize(time_limit=600)
    t1 = time.time()
    print(f"MILP solve time: {t1 - t0:.3f} s")
    _print_milp_solution(milp_model, res_milp)

    # Solve with MILP (SCIP)
    #print("\n--- Solving with MILP (SCIP) ---")
    #try:
        # Pass SCIP specific parameters if needed via scip_params
    #    scip_model = JSSPScipModel(data, scip_params={})
    #    t0 = time.time()
    #    res_scip = scip_model.optimize(time_limit=1800)
    #    t1 = time.time()
    #    print(f"SCIP solve time: {t1 - t0:.3f} s")
    #    _print_scip_solution(scip_model, res_scip)
    #except Exception as e:
    #    print(f"SCIP error: {e}")


if __name__ == '__main__':
    test_solve_ta()
