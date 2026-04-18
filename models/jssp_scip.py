from typing import Dict, List, Tuple, Any
from pyscipopt import Model, quicksum

class JSSPScipModel:
    """
    JSSP MILP model using PySCIPOpt. 
    Implements Flexible Assignment, SDST, Dual-Resource constraints, 
    Time Lags, Release Dates and Deadlines.
    """
    def __init__(self, data: Dict[str, Any], scip_params: Dict[str, Any] = None):
        self.data = data
        self.params = scip_params or {}
        self.model = Model("JSSP_MILP_SCIP")
        
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

        self.machines: List[Any] = list(d["machines"])
        self.workers: List[Any] = list(d["workers"])
        self.tasks: List[Any] = list(d["tasks"])

        # job_tasks handling
        if "job_tasks" in d:
            self.job_tasks: Dict[Any, List[Any]] = {j: list(ts) for j, ts in d["job_tasks"].items()}
            self.jobs = list(self.job_tasks.keys())
        else:
            jt = {}
            if isinstance(d["tasks"], list) and len(d["tasks"]) > 0 and isinstance(d["tasks"][0], dict) and "job" in d["tasks"][0]:
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
                if self.workers:
                    for ww in self.workers:
                        if mm in self.W_m.get(ww, []):
                            if (t, mm, ww) in self.p:
                                om_list.append((mm, ww))
                else:
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

        # V is a strict upper bound for serial execution with worst-case setups and lags
        self.V = max_r + sum_max_p + (max_s * len(self.tasks)) + (max_lag * len(self.tasks))

    def build_model(self):
        m = self.model
        
        # Disable console output for cleaner test outputs; can be toggled via scip_params
        m.hideOutput(True)
        
        # Apply SCIP specific parameters
        for k, v in self.params.items():
            try:
                m.setParam(k, v)
            except Exception:
                pass

        # Temporal variables (S_t and C_t defined as Integers for precision)
        for t in self.tasks:
            self.S[t] = m.addVar(vtype="I", lb=0.0, name=f"S_{t}")
            self.C[t] = m.addVar(vtype="I", lb=0.0, name=f"C_{t}")
        
        self.C_max = m.addVar(vtype="I", lb=0.0, name="C_max")

        has_workers = bool(self.workers)

        # Assignment binary variables (x) - defined over feasible modes OM_t
        for t in self.tasks:
            for (mm, ww) in self.OM_t.get(t, []):
                self.x[(t, mm, ww)] = m.addVar(vtype="B", name=f"x_{t}_{mm}_{ww}")

        # Machine sequencing binary variables (y)
        for i in self.tasks:
            for k in self.tasks:
                if i != k:
                    for mm in self.machines:
                        self.y[(i, k, mm)] = m.addVar(vtype="B", name=f"y_{i}_{k}_{mm}")

        # Worker sequencing binary variables (z) - only if workers exist
        if has_workers:
            for i in self.tasks:
                for k in self.tasks:
                    if i != k:
                        for ww in self.workers:
                            self.z[(i, k, ww)] = m.addVar(vtype="B", name=f"z_{i}_{k}_{ww}")

        # 1. Resource Assignment: Exactly one (m,w) pair per task
        for t in self.tasks:
            keys = [(t, mm, ww) for (mm, ww) in self.OM_t.get(t, [])]
            m.addCons(quicksum(self.x[k] for k in keys) == 1, name=f"assign_{t}")

        # 2. Completion Time: Start time + processing time of selected mode
        for t in self.tasks:
            keys = [(t, mm, ww) for (mm, ww) in self.OM_t.get(t, [])]
            expr = quicksum(self.p[(t, mm, ww)] * self.x[(t, mm, ww)] for (t, mm, ww) in keys)
            m.addCons(self.C[t] == self.S[t] + expr, name=f"comp_time_{t}")

        # 3. Precedence & Time Lags
        for (i, k) in self.P:
            lag = self.L.get((i, k), 0)
            m.addCons(self.S[k] >= self.C[i] + lag, name=f"lag_{i}_{k}")

        # 4. Job Windows (Release Dates and Deadlines)
        for j, tlist in self.job_tasks.items():
            rj = self.r.get(j, 0)
            dj = self.d.get(j, float('inf'))
            for t in tlist:
                m.addCons(self.S[t] >= rj, name=f"release_{j}_{t}")
                if dj != float('inf'):
                    m.addCons(self.C[t] <= dj, name=f"deadline_{j}_{t}")

        # 5. Machine Sequence Consistency (Linearization of logic)
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for mm in self.machines:
                    if has_workers:
                        wi = [ww for (m, ww) in self.OM_t.get(i, []) if m == mm and (i, m, ww) in self.x]
                        wk = [ww for (m, ww) in self.OM_t.get(k, []) if m == mm and (k, m, ww) in self.x]
                        sum_xi = quicksum(self.x[(i, mm, ww)] for ww in wi) if wi else 0
                        sum_xk = quicksum(self.x[(k, mm, ww)] for ww in wk) if wk else 0
                    else:
                        sum_xi = self.x[(i, mm, None)] if (i, mm, None) in self.x else 0
                        sum_xk = self.x[(k, mm, None)] if (k, mm, None) in self.x else 0

                    if (has_workers and wi) or (not has_workers):
                        m.addCons(self.y[(i, k, mm)] <= sum_xi, name=f"y_le_xi_{i}_{k}_{mm}")
                        m.addCons(self.y[(i, k, mm)] <= sum_xk, name=f"y_le_xk_{i}_{k}_{mm}")
                    else:
                        m.addCons(self.y[(i, k, mm)] <= 0, name=f"y_zero_{i}_{k}_{mm}")

                    m.addCons(self.y[(i, k, mm)] + self.y[(k, i, mm)] <= 1, name=f"y_antisym_{i}_{k}_{mm}")
                    m.addCons(
                        self.y[(i, k, mm)] + self.y[(k, i, mm)] >= sum_xi + sum_xk - 1,
                        name=f"y_force_{i}_{k}_{mm}"
                    )

        # 6. Machine Disjunction with SDST (Big-M formulation)
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for mm in self.machines:
                    y_ikm = self.y[(i, k, mm)]
                    s_ikm = self.s.get((i, k, mm), 0)
                    m.addCons(self.S[k] >= self.C[i] + s_ikm * y_ikm - self.V * (1 - y_ikm), name=f"sdst1_{i}_{k}_{mm}")
                    y_kim = self.y[(k, i, mm)]
                    s_kim = self.s.get((k, i, mm), 0)
                    m.addCons(self.S[i] >= self.C[k] + s_kim * y_kim - self.V * (1 - y_kim), name=f"sdst2_{i}_{k}_{mm}")

        # 7. Worker Sequence Consistency
        if has_workers:
            for i in self.tasks:
                for k in self.tasks:
                    if i == k:
                        continue
                    for ww in self.workers:
                        mi = [mm for (mm, w) in self.OM_t.get(i, []) if w == ww and (i, mm, w) in self.x]
                        mk = [mm for (mm, w) in self.OM_t.get(k, []) if w == ww and (k, mm, w) in self.x]
                        sum_xi_w = quicksum(self.x[(i, mm, ww)] for mm in mi) if mi else 0
                        sum_xk_w = quicksum(self.x[(k, mm, ww)] for mm in mk) if mk else 0

                        if mi:
                            m.addCons(self.z[(i, k, ww)] <= sum_xi_w, name=f"z_le_xi_{i}_{k}_{ww}")
                        else:
                            m.addCons(self.z[(i, k, ww)] <= 0, name=f"z_zero_i_{i}_{k}_{ww}")
                        if mk:
                            m.addCons(self.z[(i, k, ww)] <= sum_xk_w, name=f"z_le_xk_{i}_{k}_{ww}")
                        else:
                            m.addCons(self.z[(i, k, ww)] <= 0, name=f"z_zero_k_{i}_{k}_{ww}")

                        m.addCons(self.z[(i, k, ww)] + self.z[(k, i, ww)] <= 1, name=f"z_antisym_{i}_{k}_{ww}")
                        m.addCons(
                            self.z[(i, k, ww)] + self.z[(k, i, ww)] >= sum_xi_w + sum_xk_w - 1,
                            name=f"z_force_{i}_{k}_{ww}"
                        )

        # 8. Worker Disjunction (Big-M formulation)
        if has_workers:
            for i in self.tasks:
                for k in self.tasks:
                    if i == k:
                        continue
                    for ww in self.workers:
                        zijw = self.z[(i, k, ww)]
                        m.addCons(self.S[k] >= self.C[i] - self.V * (1 - zijw), name=f"worker_disj1_{i}_{k}_{ww}")
                        zkjw = self.z[(k, i, ww)]
                        m.addCons(self.S[i] >= self.C[k] - self.V * (1 - zkjw), name=f"worker_disj2_{i}_{k}_{ww}")

        # 9. Objective Function: Minimize Makespan
        for t in self.tasks:
            m.addCons(self.C_max >= self.C[t], name=f"makespan_{t}")

        m.setObjective(self.C_max, "minimize")

    def optimize(self, time_limit: float = None):
        """
        Runs the SCIP solver and returns a dictionary with the results.
        """
        if time_limit is not None:
            self.model.setRealParam('limits/time', float(time_limit))

        self.model.optimize()
        status = self.model.getStatus()
        
        # Mapping SCIP status strings to consistent naming
        status_map = {
            "optimal": "OPTIMAL",
            "infeasible": "INFEASIBLE",
            "unbounded": "UNBOUNDED",
            "inforunbd": "INF_OR_UNBD",
            "timelimit": "TIME_LIMIT",
            "userinterrupt": "CUTOFF",
        }
        
        status_str = status_map.get(status, f"STATUS_{status.upper()}")
        
        obj = None
        solution = {}

        # Check if at least one solution is available
        if self.model.getNSols() > 0:
            obj = self.model.getObjVal()
            
            # 1. Tasks start and end times
            tasks_sol = {}
            for t in self.tasks:
                start_val = self.model.getVal(self.S[t])
                end_val = self.model.getVal(self.C[t])
                tasks_sol[t] = {
                    "start": int(round(start_val)),
                    "end": int(round(end_val))
                }
            solution["tasks"] = tasks_sol
            
            # 2. Selected modes (Machine, Worker) per task
            selected = []
            for (t, mm, ww), var in self.x.items():
                if self.model.getVal(var) > 0.5:
                    selected.append((t, mm, ww))
            solution["selected_modes"] = selected
            
            # 3. Job windows (start of first task, end of last task)
            jobs_sol = {}
            for j, t_list in self.job_tasks.items():
                j_start = min(tasks_sol[t]["start"] for t in t_list)
                j_end = max(tasks_sol[t]["end"] for t in t_list)
                jobs_sol[j] = {"start": j_start, "end": j_end}
            solution["jobs"] = jobs_sol

        # Make solution JSON-serializable: convert tuple keys to strings and selected modes
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
                t, mm, ww = entry
                ser_sel.append([_key_to_str(t), mm, ww])
            serial_solution["selected_modes"] = ser_sel
        if solution.get("jobs"):
            serial_solution["jobs"] = { _key_to_str(j): v for j, v in solution["jobs"].items() }

        return {
            "status": status_str, 
            "scip_status": status,
            "obj_val": obj, 
            "solution": serial_solution
        }