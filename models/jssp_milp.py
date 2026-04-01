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
                # workers that can operate mm
                for ww in self.workers:
                    if mm in self.W_m.get(ww, []):
                        # ensure a processing time exists for this triple
                        if (t, mm, ww) in self.p:
                            om_list.append((mm, ww))
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

        # Build x only over feasible operation modes OM_t
        x_keys = []
        for t in self.tasks:
            for (mm, ww) in self.OM_t.get(t, []):
                x_keys.append((t, mm, ww))
        self.x = m.addVars(x_keys, vtype=GRB.BINARY, name="x")

        y_keys = []
        for i in self.tasks:
            for k in self.tasks:
                if i != k:
                    for mm in self.machines:
                        y_keys.append((i, k, mm))
        self.y = m.addVars(y_keys, vtype=GRB.BINARY, name="y")

        z_keys = []
        for i in self.tasks:
            for k in self.tasks:
                if i != k:
                    for ww in self.workers:
                        z_keys.append((i, k, ww))
        self.z = m.addVars(z_keys, vtype=GRB.BINARY, name="z")

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

        # 5. Machine Sequence Consistency
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for mm in self.machines:
                    # workers for task i that use machine mm
                    wi = [ww for (m, ww) in self.OM_t.get(i, []) if m == mm and (i, m, ww) in self.x]
                    wk = [ww for (m, ww) in self.OM_t.get(k, []) if m == mm and (k, m, ww) in self.x]

                    sum_xi = gp.quicksum(self.x[(i, mm, ww)] for ww in wi) if wi else 0
                    sum_xk = gp.quicksum(self.x[(k, mm, ww)] for ww in wk) if wk else 0

                    if wi:
                        m.addConstr(self.y[(i, k, mm)] <= sum_xi, name=f"y_le_xi_{i}_{k}_{mm}")
                    else:
                        m.addConstr(self.y[(i, k, mm)] <= 0, name=f"y_zero_i_{i}_{k}_{mm}")

                    if wk:
                        m.addConstr(self.y[(i, k, mm)] <= sum_xk, name=f"y_le_xk_{i}_{k}_{mm}")
                    else:
                        m.addConstr(self.y[(i, k, mm)] <= 0, name=f"y_zero_k_{i}_{k}_{mm}")

                    m.addConstr(self.y[(i, k, mm)] + self.y[(k, i, mm)] <= 1, name=f"y_antisym_{i}_{k}_{mm}")

                    m.addConstr(
                        self.y[(i, k, mm)] + self.y[(k, i, mm)] >= sum_xi + sum_xk - 1,
                        name=f"y_force_{i}_{k}_{mm}"
                    )

        # 6. Machine Disjunction with SDST
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for mm in self.machines:
                    y_ikm = self.y[(i, k, mm)]
                    s_ikm = self.s.get((i, k, mm), 0)
                    m.addConstr(self.S[k] >= self.C[i] + s_ikm * y_ikm - self.V * (1 - y_ikm), name=f"sdst1_{i}_{k}_{mm}")

                    y_kim = self.y[(k, i, mm)]
                    s_kim = self.s.get((k, i, mm), 0)
                    m.addConstr(self.S[i] >= self.C[k] + s_kim * y_kim - self.V * (1 - y_kim), name=f"sdst2_{i}_{k}_{mm}")

        # 7. Worker Sequence Consistency
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
        if time_limit is not None:
            try:
                self.model.setParam(GRB.Param.TimeLimit, float(time_limit))
            except Exception:
                pass

        self.model.optimize()
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
        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.CUTOFF):
            try:
                obj = self.model.objVal
            except Exception:
                obj = None

        return {"status": status_str, "gurobi_status": status, "obj_val": obj}