"""
JSSP MILP model using highspy. 
Implements Flexible Assignment, SDST, Dual-Resource constraints, 
Time Lags, Release Dates and Deadlines.
"""
from typing import Dict, List, Tuple, Any
import highspy
import numpy as np

class JSSPHighsModel:
    def __init__(self, data: Dict[str, Any], highs_params: Dict[str, Any] = None):
        self.data = data
        self.params = highs_params or {}
        self.h = highspy.Highs()
        
        # Dictionaries to store variable indices
        self.S = {}
        self.C = {}
        self.C_max = None
        self.x = {}
        self.y = {}
        self.z = {}

        self._validate_and_parse_data()
        self.build_model()

    def _validate_and_parse_data(self):
        d = self.data
        required = ["machines", "workers", "P", "M_t", "W_m", "p", "tasks"]
        for k in required:
            if k not in d:
                raise KeyError(f"Data dictionary must contain key '{k}'")

        self.machines = list(d["machines"])
        self.workers = list(d["workers"])
        self.tasks = list(d["tasks"])

        # job_tasks mapping
        if "job_tasks" in d:
            self.job_tasks = {j: list(ts) for j, ts in d["job_tasks"].items()}
        else:
            # Fallback for data structures where job is an attribute of the task
            jt = {}
            for t in d["tasks"]:
                tid = t.get("id") if isinstance(t, dict) else t
                job = t["job"] if isinstance(t, dict) else "default_job"
                jt.setdefault(job, []).append(tid)
            self.job_tasks = jt

        self.P = list(d.get("P", []))
        self.L = {tuple(k): int(round(float(v))) for k, v in d.get("L", {}).items()} if d.get("L") else {}
        self.r = {j: int(round(float(v))) for j, v in d.get("release_dates", {}).items()} if d.get("release_dates") else {}
        self.d = {j: int(round(float(v))) for j, v in d.get("deadlines", {}).items()} if d.get("deadlines") else {}
        self.M_t = {t: list(ms) for t, ms in d["M_t"].items()}
        self.W_m = {w: list(ms) for w, ms in d["W_m"].items()}

        # Normalize processing times
        self.p = {}
        for key, value in d["p"].items():
            k = tuple(key)
            self.p[k] = int(round(float(value)))

        self.s = {tuple(k): int(round(float(v))) for k, v in d["s"].items()} if d.get("s") else {}

        # Feasible operation modes (machine, worker) per task
        self.OM_t = {}
        for t in self.tasks:
            modes = []
            for mm in self.M_t.get(t, []):
                if self.workers:
                    for ww in self.workers:
                        if mm in self.W_m.get(ww, []) and (t, mm, ww) in self.p:
                            modes.append((mm, ww))
                else:
                    if (t, mm, None) in self.p:
                        modes.append((mm, None))
            self.OM_t[t] = modes

        # Big-M (V) calculation
        max_r = max(self.r.values()) if self.r else 0
        sum_p = sum(max([self.p.get((t, m, w), 0) for m, w in self.OM_t[t]], default=0) for t in self.tasks)
        max_s = max(self.s.values()) if self.s else 0
        max_l = max(self.L.values()) if self.L else 0
        self.V = max_r + sum_p + (max_s * len(self.tasks)) + (max_l * len(self.tasks))

    def _add_var(self, name: str, vtype: str, lb: float = 0.0, ub: float = None):
        """
        Helper to add a variable to HiGHS.
        Fixes the 'incompatible function arguments' error by using 
        individual property setters.
        """
        upper = ub if ub is not None else highspy.kHighsInf
        if vtype == "B":
            upper = 1.0
            
        # 1. Add a continuous variable with bounds first
        # This matches the (lb, ub, obj, name, type) or simple (lb, ub) signatures
        self.h.addVar(lb, upper)
        
        # 2. Get the index of the variable we just added
        var_idx = self.h.getNumCol() - 1
        
        # 3. Set the variable type separately
        if vtype in ["I", "B"]:
            self.h.changeColIntegrality(var_idx, highspy.HighsVarType.kInteger)
        else:
            self.h.changeColIntegrality(var_idx, highspy.HighsVarType.kContinuous)
            
        # 4. Set the name if needed (optional for solving, good for debugging)
        # Note: some highspy versions use different methods for naming
        # self.h.changeColName(var_idx, name) 
        
        return var_idx

    def _add_row(self, lb: float, ub: float, indices: List[int], coeffs: List[float]):
        """
        Helper to add a row (constraint) to HiGHS.
        Ensures types match the expected (lb, ub, len, indices, coeffs) signature.
        """
        if not indices:
            return
            
        # HiGHS requires NumPy arrays for the indices and values
        # Argument order: lower_bound, upper_bound, num_non_zeros, indices, values
        self.h.addRow(
            float(lb), 
            float(ub), 
            len(indices), 
            np.array(indices, dtype=np.int32), 
            np.array(coeffs, dtype=np.float64)
        )

    def build_model(self):
        h = self.h
        # Set solver parameters (e.g., output_flag)
        h.setOptionValue('output_flag', False)
        for k, v in self.params.items():
            h.setOptionValue(k, v)

        # 1. Variables
        for t in self.tasks:
            self.S[t] = self._add_var(f"S_{t}", "I")
            self.C[t] = self._add_var(f"C_{t}", "I")
        
        self.C_max = self._add_var("C_max", "I")

        has_workers = bool(self.workers)
        for t in self.tasks:
            for (mm, ww) in self.OM_t[t]:
                self.x[(t, mm, ww)] = self._add_var(f"x_{t}_{mm}_{ww}", "B")

        for i in self.tasks:
            for k in self.tasks:
                if i != k:
                    for mm in self.machines:
                        self.y[(i, k, mm)] = self._add_var(f"y_{i}_{k}_{mm}", "B")
                    if has_workers:
                        for ww in self.workers:
                            self.z[(i, k, ww)] = self._add_var(f"z_{i}_{k}_{ww}", "B")

        # 2. Constraints
        # HiGHS uses addRow(lower_bound, upper_bound, [indices], [coefficients])
        
        # 2.1 Assignment: sum(x) == 1
        for t in self.tasks:
            keys = [(t, mm, ww) for (mm, ww) in self.OM_t[t]]
            indices = [self.x[k] for k in keys]
            coeffs = [1.0] * len(indices)
            self._add_row(1.0, 1.0, indices, coeffs)

        # 2.2 Completion Time: C[t] - S[t] - sum(p*x) == 0
        for t in self.tasks:
            keys = [(t, mm, ww) for (mm, ww) in self.OM_t[t]]
            indices = [self.C[t], self.S[t]] + [self.x[k] for k in keys]
            coeffs = [1.0, -1.0] + [-float(self.p[k]) for k in keys]
            self._add_row(0.0, 0.0, indices, coeffs)

        # 2.3 Precedence & Lags: S[k] - C[i] >= lag
        for (i, k) in self.P:
            lag = self.L.get((i, k), 0)
            self._add_row(float(lag), highspy.kHighsInf, [self.S[k], self.C[i]], [1.0, -1.0])

        # 2.4 Job Windows (Release/Deadline)
        for j, tlist in self.job_tasks.items():
            rj = self.r.get(j, 0)
            dj = self.d.get(j, float('inf'))
            for t in tlist:
                # Release date: S[t] >= rj
                self._add_row(float(rj), highspy.kHighsInf, [self.S[t]], [1.0])
                if dj != float('inf'):
                    # Deadline: C[t] <= dj
                    self._add_row(-highspy.kHighsInf, float(dj), [self.C[t]], [1.0])

        # 2.5 Machine Sequencing Consistency
        for i in self.tasks:
            for k in self.tasks:
                if i == k: continue
                for mm in self.machines:
                    # Filter feasible modes for machine mm
                    modes_i = [m for m in self.OM_t[i] if m[0] == mm]
                    modes_k = [m for m in self.OM_t[k] if m[0] == mm]
                    
                    # Logic: y_ikm <= sum(x_i) AND y_ikm <= sum(x_k)
                    xi_indices = [self.x[(i, m, w)] for m, w in modes_i]
                    xk_indices = [self.x[(k, m, w)] for m, w in modes_k]
                    
                    if modes_i and modes_k:
                        y_idx = self.y[(i, k, mm)]
                        # y_ikm - sum(x_i) <= 0
                        self._add_row(-highspy.kHighsInf, 0.0, [y_idx] + xi_indices, [1.0] + [-1.0]*len(xi_indices))
                        # y_ikm - sum(x_k) <= 0
                        self._add_row(-highspy.kHighsInf, 0.0, [y_idx] + xk_indices, [1.0] + [-1.0]*len(xk_indices))
                        
                        # Antisymmetry: y_ikm + y_kim <= 1
                        self._add_row(0.0, 1.0, [self.y[(i, k, mm)], self.y[(k, i, mm)]], [1.0, 1.0])
                        
                        # Force sequence: y_ikm + y_kim >= sum(x_i) + sum(x_k) - 1
                        # y_ikm + y_kim - sum(x_i) - sum(x_k) >= -1
                        idx_force = [self.y[(i, k, mm)], self.y[(k, i, mm)]] + xi_indices + xk_indices
                        val_force = [1.0, 1.0] + [-1.0]*len(xi_indices) + [-1.0]*len(xk_indices)
                        self._add_row(-1.0, highspy.kHighsInf, idx_force, val_force)
                    else:
                        # If machine is not feasible for both tasks, sequence variable must be 0
                        self._add_row(0.0, 0.0, [self.y[(i, k, mm)]], [1.0])

        # 2.6 Machine Disjunction with SDST (Big-M)
        for i in self.tasks:
            for k in self.tasks:
                if i == k: continue
                for mm in self.machines:
                    if (i, k, mm) in self.y:
                        s_ikm = self.s.get((i, k, mm), 0)
                        # S[k] >= C[i] + s*y - V*(1-y)  => S[k] - C[i] - (s+V)y >= -V
                        idx = [self.S[k], self.C[i], self.y[(i, k, mm)]]
                        val = [1.0, -1.0, -float(s_ikm + self.V)]
                        self._add_row(-float(self.V), highspy.kHighsInf, idx, val)

        # 2.7 Worker Sequencing Consistency & Disjunction
        if has_workers:
            for i in self.tasks:
                for k in self.tasks:
                    if i == k: continue
                    for ww in self.workers:
                        modes_i = [m for m in self.OM_t[i] if m[1] == ww]
                        modes_k = [m for m in self.OM_t[k] if m[1] == ww]
                        
                        if modes_i and modes_k:
                            z_idx = self.z[(i, k, ww)]
                            xi_w = [self.x[(i, m, w)] for m, w in modes_i]
                            xk_w = [self.x[(k, m, w)] for m, w in modes_k]
                            
                            # z_ikw <= sum(x_i_w) and z_ikw <= sum(x_k_w)
                            self._add_row(-highspy.kHighsInf, 0.0, [z_idx] + xi_w, [1.0] + [-1.0]*len(xi_w))
                            self._add_row(-highspy.kHighsInf, 0.0, [z_idx] + xk_w, [1.0] + [-1.0]*len(xk_w))
                            
                            # Antisymmetry for workers: z_ikw + z_kiw <= 1
                            self._add_row(0.0, 1.0, [self.z[(i, k, ww)], self.z[(k, i, ww)]], [1.0, 1.0])
                            
                            # Force sequence logic for workers
                            idx_f = [self.z[(i, k, ww)], self.z[(k, i, ww)]] + xi_w + xk_w
                            val_f = [1.0, 1.0] + [-1.0]*len(xi_w) + [-1.0]*len(xk_w)
                            self._add_row(-1.0, highspy.kHighsInf, idx_f, val_f)
                            
                            # Worker Disjunction: S[k] - C[i] + V*z >= 0 (Simplified since s=0 for workers)
                            # S[k] - C[i] - V*(1-z) >= 0 => S[k] - C[i] + V*z >= V
                            # Note: Adjusted Big-M formulation to match machine disjunction style
                            self._add_row(-float(self.V), highspy.kHighsInf, [self.S[k], self.C[i], z_idx], [1.0, -1.0, -float(self.V)])
                        else:
                             # If worker is not feasible for both tasks, sequence variable must be 0
                             self._add_row(0.0, 0.0, [self.z[(i, k, ww)]], [1.0])

        # 2.8 Makespan
        for t in self.tasks:
            # C_max >= C[t] => C_max - C[t] >= 0
            self._add_row(0.0, highspy.kHighsInf, [self.C_max, self.C[t]], [1.0, -1.0])

        # 3. Objective: Minimize Makespan
        # Use changeColCost to set the coefficient for C_max in the objective
        h.changeColCost(self.C_max, 1.0)
        # h.changeModelStatus(highspy.HighsModelStatus.kNotset)

    def optimize(self, time_limit: float = None):
        if time_limit is not None:
            self.h.setOptionValue('time_limit', float(time_limit))
        
        self.h.run()
        
        # Get results
        model_status = self.h.getModelStatus()
        info = self.h.getInfo()
        
        status_map = {
            highspy.HighsModelStatus.kOptimal: "OPTIMAL",
            highspy.HighsModelStatus.kInfeasible: "INFEASIBLE",
            highspy.HighsModelStatus.kUnbounded: "UNBOUNDED",
            highspy.HighsModelStatus.kTimeLimit: "TIME_LIMIT",
        }
        
        status_str = status_map.get(model_status, f"STATUS_{model_status}")
        obj_val = info.objective_function_value if status_str in ["OPTIMAL", "TIME_LIMIT"] else None
        
        return {
            "status": status_str,
            "highs_status": model_status,
            "obj_val": obj_val
        }