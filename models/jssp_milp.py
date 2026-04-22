"""
JSSP MILP model using gurobipy. Implements Flexible Assignment, SDST, Dual-Resource
constraints, Time Lags, Release Dates and Deadlines.

Class: JSSPMilpModel
- __init__(data, gurobi_params=None): stores data and builds model
- build_model(): constructs variables and constraints
- optimize(): runs solver and returns status and objective value
"""
from typing import Dict, List, Tuple, Any
import gurobipy as gp
from gurobipy import GRB


class JSSPMilpModel:
    def __init__(self, data: Dict[str, Any], gurobi_params: Dict[str, Any] = None):
        self.data = data
        self.params = gurobi_params or {}
        self.model = gp.Model("JSSP_MILP")
        
        self.S = None
        self.C = None
        self.C_max = None
        self.x = None
        self.y = None
        self.z = None

        self._validate_and_parse_data()
        self.build_model()

    def _validate_and_parse_data(self):
        d = self.data

        required = ["machines", "workers", "P", "M_t", "W_m", "p", "tasks"]
        for k in required:
            if k not in d:
                raise KeyError(f"Data dictionary must contain key '{k}'")

        self.machines: List[Any] = list(d["machines"])
        self.workers: List[Any] = list(d["workers"])
        self.tasks: List[Any] = list(d["tasks"])

        # job_tasks handling
        if "job_tasks" in d:
            self.job_tasks: Dict[Any, List[Any]] = {j: list(ts) for j, ts in d["job_tasks"].items()}
            self.jobs = list(self.job_tasks.keys())
        else:
            jt = {}
            if isinstance(d["tasks"], list) and len(d["tasks"])>0 and isinstance(d["tasks"][0], dict) and "job" in d["tasks"][0]:
                for t in d["tasks"]:
                    tid = t.get("id") if t.get("id") is not None else t
                    job = t["job"]
                    jt.setdefault(job, []).append(tid)
                self.job_tasks = jt
                self.jobs = list(jt.keys())
                self.tasks = [t.get("id") if isinstance(t, dict) else t for t in d["tasks"]]
            else:
                raise KeyError("Data must provide 'job_tasks' or tasks must include job membership")

        # precedence and lags
        self.P: List[Tuple[Any, Any]] = list(d.get("P", []))

        # Use int() for temporal parameters to ensure numerical precision
        self.L: Dict[Tuple[Any, Any], int] = {tuple(k): int(round(float(v))) for k, v in d.get("L", {}).items()} if d.get("L") else {}
        self.r: Dict[Any, int] = {j: int(round(float(v))) for j, v in d.get("release_dates", {}).items()} if d.get("release_dates") else {}
        self.d: Dict[Any, int] = {j: int(round(float(v))) for j, v in d.get("deadlines", {}).items()} if d.get("deadlines") else {}

        # eligibility: machines per task
        self.M_t: Dict[Any, List[Any]] = {t: list(ms) for t, ms in d["M_t"].items()}
        # worker->machines capability matrix
        self.W_m: Dict[Any, List[Any]] = {w: list(ms) for w, ms in d["W_m"].items()}

        # processing times
        self.p: Dict[Tuple[Any, Any, Any], int] = {}
        for key, value in d["p"].items():
            if not self.workers:
                normalized_key = (key[0], key[1], None)
                self.p[normalized_key] = int(round(float(value)))
            else:
                self.p[tuple(key)] = int(round(float(value)))

        # setups
        self.s: Dict[Tuple[Any, Any, Any], int] = {}
        if d.get("s"):
            for key, value in d["s"].items():
                self.s[tuple(key)] = int(round(float(value)))

        # --- CONSTRUCT OM_t: feasible operation modes (machine, worker) per task ---
        self.OM_t: Dict[Any, List[Tuple[Any, Any]]] = {}
        for t in self.tasks:
            om_list = []
            for mm in self.M_t.get(t, []):
                # if workers are provided, match workers that can operate mm
                if self.workers:
                    for ww in self.workers:
                        if mm in self.W_m.get(ww, []):
                            if (t, mm, ww) in self.p:
                                om_list.append((mm, ww))
                else:
                    # no workers: check directly if (task, mm, None) exists in normalized p
                    if (t, mm, None) in self.p:
                        om_list.append((mm, None))
            self.OM_t[t] = om_list
            if not om_list:
                raise ValueError(f"Task {t} has no feasible (machine,worker) modes in OM_t")

        # --- DYNAMIC CALCULATION OF BIG-M (V) ---
        max_r = max(self.r.values()) if self.r else 0
        sum_max_p = 0
        for t in self.tasks:
            max_p_t = max([self.p.get((t, mm, ww), 0) for (mm, ww) in self.OM_t.get(t, [])], default=0)
            sum_max_p += max_p_t

        max_s = max(self.s.values()) if self.s else 0
        max_lag = max(self.L.values()) if self.L else 0

        # V is a strict upper bound: if everything is executed serially with worst setups and lags
        self.V = max_r + sum_max_p + (max_s * len(self.tasks)) + (max_lag * len(self.tasks))

    def build_model(self):
        m = self.model
        m.setParam('OutputFlag', 0)
        for k, v in self.params.items():
            try:
                m.setParam(k, v)
            except Exception:
                pass

        self.S = m.addVars(self.tasks, vtype=GRB.INTEGER, lb=0.0, name="S")
        self.C = m.addVars(self.tasks, vtype=GRB.INTEGER, lb=0.0, name="C")
        self.C_max = m.addVar(vtype=GRB.INTEGER, lb=0.0, name="C_max")

        has_workers = bool(self.workers)

        # Build x only over feasible operation modes OM_t
        x_keys = []
        for t in self.tasks:
            for (mm, ww) in self.OM_t.get(t, []):
                # use None for worker dimension when no workers exist
                x_keys.append((t, mm, ww))
        self.x = m.addVars(x_keys, vtype=GRB.BINARY, name="x")

        # y variables (machine sequencing) always needed
        y_keys = []
        for i in self.tasks:
            for k in self.tasks:
                if i != k:
                    for mm in self.machines:
                        y_keys.append((i, k, mm))
        self.y = m.addVars(y_keys, vtype=GRB.BINARY, name="y")

        # z variables (worker sequencing) only if workers exist
        if has_workers:
            z_keys = []
            for i in self.tasks:
                for k in self.tasks:
                    if i != k:
                        for ww in self.workers:
                            z_keys.append((i, k, ww))
            self.z = m.addVars(z_keys, vtype=GRB.BINARY, name="z")
        else:
            self.z = None

        m.update()

        # 1. Resource Assignment: enforce exactly one feasible (m,w) per task
        for t in self.tasks:
            keys = [(t, mm, ww) for (mm, ww) in self.OM_t.get(t, [])]
            if not keys:
                raise ValueError(f"No eligible (machine,worker) pairs for task {t}")
            m.addConstr(gp.quicksum(self.x[k] for k in keys) == 1, name=f"assign_{t}")

        # 2. Completion Time: sum only over OM_t
        for t in self.tasks:
            keys = [(t, mm, ww) for (mm, ww) in self.OM_t.get(t, [])]
            expr = gp.quicksum(self.p[(t, mm, ww)] * self.x[(t, mm, ww)] for (t, mm, ww) in keys)
            m.addConstr(self.C[t] == self.S[t] + expr, name=f"comp_time_{t}")

        # 3. Precedence & Time Lags
        for (i, k) in self.P:
            lag = self.L.get((i, k), 0)
            m.addConstr(self.S[k] >= self.C[i] + lag, name=f"lag_{i}_{k}")

        # 4. Job Windows
        for j, tlist in self.job_tasks.items():
            rj = self.r.get(j, 0)
            dj = self.d.get(j, float('inf'))
            for t in tlist:
                m.addConstr(self.S[t] >= rj, name=f"release_{j}_{t}")
                if dj != float('inf'):
                    m.addConstr(self.C[t] <= dj, name=f"deadline_{j}_{t}")

        # If not SDST constraints, standard y_ik definition
        if not self.s:

            # 5. Machine Sequence Consistency
            for i in self.tasks:
                for k in self.tasks:
                    if i == k:
                        continue
                    for mm in self.machines:
                        # workers for task i that use machine mm
                        if has_workers:
                            wi = [ww for (m, ww) in self.OM_t.get(i, []) if m == mm and (i, m, ww) in self.x]
                            wk = [ww for (m, ww) in self.OM_t.get(k, []) if m == mm and (k, m, ww) in self.x]

                            sum_xi = gp.quicksum(self.x[(i, mm, ww)] for ww in wi) if wi else 0
                            sum_xk = gp.quicksum(self.x[(k, mm, ww)] for ww in wk) if wk else 0
                        else:
                            # no workers: x keys use ww=None
                            sum_xi = self.x[(i, mm, None)] if (i, mm, None) in self.x else 0
                            sum_xk = self.x[(k, mm, None)] if (k, mm, None) in self.x else 0

                        if has_workers and wi:
                            m.addConstr(self.y[(i, k, mm)] <= sum_xi, name=f"y_le_xi_{i}_{k}_{mm}")
                        elif not has_workers:
                            # when no workers exist, y is tied to x presence on machine mm
                            m.addConstr(self.y[(i, k, mm)] <= sum_xi, name=f"y_le_xi_{i}_{k}_{mm}")
                        else:
                            m.addConstr(self.y[(i, k, mm)] <= 0, name=f"y_zero_i_{i}_{k}_{mm}")

                        if has_workers and wk:
                            m.addConstr(self.y[(i, k, mm)] <= sum_xk, name=f"y_le_xk_{i}_{k}_{mm}")
                        elif not has_workers:
                            m.addConstr(self.y[(i, k, mm)] <= sum_xk, name=f"y_le_xk_{i}_{k}_{mm}")
                        else:
                            m.addConstr(self.y[(i, k, mm)] <= 0, name=f"y_zero_k_{i}_{k}_{mm}")

                        m.addConstr(self.y[(i, k, mm)] + self.y[(k, i, mm)] <= 1, name=f"y_antisym_{i}_{k}_{mm}")

                        m.addConstr(
                            self.y[(i, k, mm)] + self.y[(k, i, mm)] >= sum_xi + sum_xk - 1,
                            name=f"y_force_{i}_{k}_{mm}"
                        )
        # If SDST constraints are present y_ik vars represent direct sequencing
        else:
            # --- Helper Expressions for Routing Formulation ---
            # X_tm[t, m] = 1 if task t is assigned to machine m
            X_tm = {}
            for t in self.tasks:
                for mm in self.machines:
                    if has_workers:
                        valid_w = [ww for (m, ww) in self.OM_t.get(t, []) if m == mm]
                        X_tm[(t, mm)] = gp.quicksum(self.x[(t, mm, ww)] for ww in valid_w) if valid_w else 0
                    else:
                        X_tm[(t, mm)] = self.x[(t, mm, None)] if (t, mm, None) in self.x else 0
                        
            # 5. Machine Sequence Consistency (Routing Formulation)
            for mm in self.machines:
                # Path forcing: total connections >= assigned tasks - 1
                sum_y = gp.quicksum(self.y[(i, k, mm)] for i in self.tasks for k in self.tasks if i != k)
                sum_X = gp.quicksum(X_tm[(t, mm)] for t in self.tasks)
                m.addConstr(sum_y >= sum_X - 1, name=f"path_forcing_m_{mm}")

                for i in self.tasks:
                    # Out-degree <= 1
                    m.addConstr(gp.quicksum(self.y[(i, k, mm)] for k in self.tasks if i != k) <= X_tm[(i, mm)], name=f"out_degree_{i}_{mm}")
                    
                    # In-degree <= 1
                    m.addConstr(gp.quicksum(self.y[(k, i, mm)] for k in self.tasks if i != k) <= X_tm[(i, mm)], name=f"in_degree_{i}_{mm}")

                    for k in self.tasks:
                        if i != k:
                            # Antisymmetry
                            m.addConstr(self.y[(i, k, mm)] + self.y[(k, i, mm)] <= 1, name=f"antisym_{i}_{k}_{mm}")


        # 6. Machine Disjunction with SDST
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for mm in self.machines:
                    y_ikm = self.y[(i, k, mm)]
                    s_ikm = self.s.get((i, k, mm), 0)
                    m.addConstr(self.S[k] >= self.C[i] + s_ikm - self.V * (1 - y_ikm), name=f"sdst1_{i}_{k}_{mm}")

                    y_kim = self.y[(k, i, mm)]
                    s_kim = self.s.get((k, i, mm), 0)
                    m.addConstr(self.S[i] >= self.C[k] + s_kim - self.V * (1 - y_kim), name=f"sdst2_{i}_{k}_{mm}")

        # 7. Worker Sequence Consistency
        if has_workers:
            for i in self.tasks:
                for k in self.tasks:
                    if i == k:
                        continue
                    for ww in self.workers:
                        # machines for task i that can be operated by ww
                        mi = [mm for (mm, w) in self.OM_t.get(i, []) if w == ww and (i, mm, w) in self.x]
                        mk = [mm for (mm, w) in self.OM_t.get(k, []) if w == ww and (k, mm, w) in self.x]

                        sum_xi_w = gp.quicksum(self.x[(i, mm, ww)] for mm in mi) if mi else 0
                        sum_xk_w = gp.quicksum(self.x[(k, mm, ww)] for mm in mk) if mk else 0

                        if mi:
                            m.addConstr(self.z[(i, k, ww)] <= sum_xi_w, name=f"z_le_xi_{i}_{k}_{ww}")
                        else:
                            m.addConstr(self.z[(i, k, ww)] <= 0, name=f"z_zero_i_{i}_{k}_{ww}")

                        if mk:
                            m.addConstr(self.z[(i, k, ww)] <= sum_xk_w, name=f"z_le_xk_{i}_{k}_{ww}")
                        else:
                            m.addConstr(self.z[(i, k, ww)] <= 0, name=f"z_zero_k_{i}_{k}_{ww}")

                        m.addConstr(self.z[(i, k, ww)] + self.z[(k, i, ww)] <= 1, name=f"z_antisym_{i}_{k}_{ww}")

                        m.addConstr(
                            self.z[(i, k, ww)] + self.z[(k, i, ww)] >= sum_xi_w + sum_xk_w - 1,
                            name=f"z_force_{i}_{k}_{ww}"
                        )

        # 8. Worker Disjunction
        if has_workers:
            for i in self.tasks:
                for k in self.tasks:
                    if i == k:
                        continue
                    for ww in self.workers:
                        zijw = self.z[(i, k, ww)]
                        m.addConstr(self.S[k] >= self.C[i] - self.V * (1 - zijw), name=f"worker_disj1_{i}_{k}_{ww}")
                        zkjw = self.z[(k, i, ww)]
                        m.addConstr(self.S[i] >= self.C[k] - self.V * (1 - zkjw), name=f"worker_disj2_{i}_{k}_{ww}")

        # 9. Makespan
        for t in self.tasks:
            m.addConstr(self.C_max >= self.C[t], name=f"makespan_{t}")

        m.setObjective(self.C_max, GRB.MINIMIZE)
        m.update()

    def optimize(self, time_limit: float = None):
        """
        Runs the Gurobi solver and returns a dictionary with the results.
        """
        if time_limit is not None:
            try:
                self.model.setParam(GRB.Param.TimeLimit, float(time_limit))
            except Exception:
                pass

        self.model.optimize()
        
        # Mapping Gurobi status codes to consistent string naming
        status = self.model.status
        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.CUTOFF: "CUTOFF",
        }
        status_str = status_map.get(status, f"STATUS_{status}")
        
        obj = None
        solution = {}

        # Check if Gurobi found at least one feasible solution
        if self.model.SolCount > 0:
            try:
                obj = self.model.objVal
                
                # 1. Extract task start and end times (rounded for precision)
                tasks_sol = {}
                for t in self.tasks:
                    tasks_sol[t] = {
                        "start": int(round(self.S[t].X)),
                        "end": int(round(self.C[t].X))
                    }
                solution["tasks"] = tasks_sol
                
                # 2. Extract selected operation modes (Machine, Worker) per task
                selected = []
                for (t, mm, ww), var in self.x.items():
                    # If binary variable is active (True)
                    if var.X > 0.5:
                        selected.append((t, mm, ww))
                solution["selected_modes"] = selected
                
                # 3. Calculate job windows (first task start, last task end)
                jobs_sol = {}
                for j, t_list in self.job_tasks.items():
                    # Job starts when its first task starts and ends with its last task
                    j_start = min(tasks_sol[t]["start"] for t in t_list)
                    j_end = max(tasks_sol[t]["end"] for t in t_list)
                    jobs_sol[j] = {"start": j_start, "end": j_end}
                solution["jobs"] = jobs_sol
                
            except Exception:
                # In case of numeric issues during attribute access
                obj = None
                solution = {}
        def _key_to_str(k):
            if isinstance(k, tuple):
                return "_".join(str(x) for x in k)
            return str(k)

        serial_solution = {}
        if solution.get("tasks"):
            serial_solution["tasks"] = { _key_to_str(t): v for t, v in solution["tasks"].items() }
        if solution.get("selected_modes") is not None:
            ser_sel = []
            for entry in solution.get("selected_modes", []):
                t, m, w = entry
                ser_sel.append([_key_to_str(t), m, w])
            serial_solution["selected_modes"] = ser_sel
        if solution.get("jobs"):
            serial_solution["jobs"] = { _key_to_str(j): v for j, v in solution["jobs"].items() }
    

        return {
            "status": status_str, 
            "gurobi_status": status, 
            "obj_val": obj, 
            "solution": serial_solution
        }