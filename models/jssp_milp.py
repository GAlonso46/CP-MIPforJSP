"""
JSSP MILP model using gurobipy following the exact mathematical formulation
provided by the user. Implements Flexible Assignment, SDST, Dual-Resource
constraints, Time Lags, Release Dates and Deadlines.

Class: JSSPMilpModel
- __init__(data, gurobi_params=None): stores data and builds model
- build_model(): constructs variables and constraints
- optimize(): runs solver and returns status and objective value

The data dictionary must contain the necessary keys. The constructor will
attempt to be permissive about input formats but will raise informative
errors if required pieces are missing.

This module intentionally implements only the constraints described in the
formulation; no extra constraints are added.
"""
from typing import Dict, List, Tuple, Any
import gurobipy as gp
from gurobipy import GRB


class JSSPMilpModel:
    """Builds a Gurobi model for the extended JSSP described by the user.

    Expected (recommended) keys in data dictionary:
      - jobs: List of job ids
      - tasks: List of task ids
      - job_tasks: Dict[job, List[task]] OR tasks can be objects with 'job' attribute
      - machines: List of machine ids
      - workers: List of worker ids
      - P: List of precedence arcs as tuples (i,k)
      - L: Dict[(i,k), lag] optional (defaults to 0)
      - release_dates: Dict[job, r_j] optional (defaults to 0)
      - deadlines: Dict[job, d_j] optional (defaults to +inf)
      - M_t: Dict[task, List[machine]] eligible machines for task
      - W_t: Dict[task, List[worker]] eligible workers for task
      - p: Dict[(t,m,w), processing_time]
      - s: Dict[(i,k,m), setup_time] sequence-dependent setup on machine m
      - V: Big-M constant (required)

    The class does not invent constraints outside the provided formulation.
    """

    def __init__(self, data: Dict[str, Any], gurobi_params: Dict[str, Any] = None):
        self.data = data
        self.params = gurobi_params or {}
        self.model = gp.Model("JSSP_MILP")
        # will be populated in build
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
        # required top-level keys
        required = ["machines", "workers", "P", "M_t", "W_t", "p", "V", "tasks"]
        for k in required:
            if k not in d:
                raise KeyError(f"Data dictionary must contain key '{k}'")

        # sets
        self.machines: List[Any] = list(d["machines"])
        self.workers: List[Any] = list(d["workers"])

        # tasks
        self.tasks: List[Any] = list(d["tasks"])

        # jobs <-> tasks mapping
        if "job_tasks" in d:
            self.job_tasks: Dict[Any, List[Any]] = {j: list(ts) for j, ts in d["job_tasks"].items()}
            self.jobs = list(self.job_tasks.keys())
        else:
            # try to infer: tasks may be list of dicts with 'job'
            jt = {}
            if isinstance(d["tasks"], list) and len(d["tasks"])>0 and isinstance(d["tasks"][0], dict) and "job" in d["tasks"][0]:
                for t in d["tasks"]:
                    tid = t.get("id") if t.get("id") is not None else t
                    job = t["job"]
                    jt.setdefault(job, []).append(tid)
                self.job_tasks = jt
                self.jobs = list(jt.keys())
                # normalize tasks list to ids
                self.tasks = [t.get("id") if isinstance(t, dict) else t for t in d["tasks"]]
            else:
                raise KeyError("Data must provide 'job_tasks' or tasks must include job membership")

        # precedence arcs
        self.P: List[Tuple[Any, Any]] = list(d.get("P", []))
        # time lags
        self.L: Dict[Tuple[Any, Any], float] = {tuple(k): float(v) for k, v in d.get("L", {}).items()} if d.get("L") else {}
        # release dates and deadlines
        self.r: Dict[Any, float] = {j: float(v) for j, v in d.get("release_dates", {}).items()} if d.get("release_dates") else {}
        self.d: Dict[Any, float] = {j: float(v) for j, v in d.get("deadlines", {}).items()} if d.get("deadlines") else {}

        # eligibility
        self.M_t: Dict[Any, List[Any]] = {t: list(ms) for t, ms in d["M_t"].items()}
        self.W_t: Dict[Any, List[Any]] = {t: list(ws) for t, ws in d["W_t"].items()}

        # processing times
        # expect keys as (t,m,w)
        self.p: Dict[Tuple[Any, Any, Any], float] = {}
        for key, value in d["p"].items():
            self.p[tuple(key)] = float(value)

        # setup times s[(i,k,m)]
        self.s: Dict[Tuple[Any, Any, Any], float] = {}
        if d.get("s"):
            for key, value in d["s"].items():
                self.s[tuple(key)] = float(value)

        # Big-M
        self.V: float = float(d["V"]) if "V" in d else None
        if self.V is None:
            raise KeyError("Big-M parameter 'V' must be provided in data as key 'V'")

    def build_model(self):
        m = self.model
        m.setParam('OutputFlag', 0)  # silent by default
        for k, v in self.params.items():
            try:
                m.setParam(k, v)
            except Exception:
                # ignore invalid params
                pass

        # Decision variables
        # Start and completion times for each task
        self.S = m.addVars(self.tasks, vtype=GRB.CONTINUOUS, lb=0.0, name="S")
        self.C = m.addVars(self.tasks, vtype=GRB.CONTINUOUS, lb=0.0, name="C")
        self.C_max = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="C_max")

        # x_{t,m,w}
        # create x only for eligible (m,w) pairs of each task
        x_keys = []
        for t in self.tasks:
            ms = self.M_t.get(t, [])
            ws = self.W_t.get(t, [])
            for mm in ms:
                for ww in ws:
                    x_keys.append((t, mm, ww))
        self.x = m.addVars(x_keys, vtype=GRB.BINARY, name="x")

        # y_{i,k}^m for all ordered task pairs and machines
        y_keys = []
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for mm in self.machines:
                    y_keys.append((i, k, mm))
        self.y = m.addVars(y_keys, vtype=GRB.BINARY, name="y")

        # z_{i,k}^w for all ordered task pairs and workers
        z_keys = []
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for ww in self.workers:
                    z_keys.append((i, k, ww))
        self.z = m.addVars(z_keys, vtype=GRB.BINARY, name="z")

        m.update()

        # 1. Resource Assignment: sum_{m in M_t, w in W_t} x_{t,m,w} == 1
        for t in self.tasks:
            keys = [(t, mm, ww) for mm in self.M_t.get(t, []) for ww in self.W_t.get(t, [])]
            if not keys:
                raise ValueError(f"No eligible (machine,worker) pairs for task {t}")
            m.addConstr(gp.quicksum(self.x[k] for k in keys) == 1, name=f"assign_{t}")

        # 2. Completion Time: C_t = S_t + sum(p_{t,m,w} * x_{t,m,w})
        for t in self.tasks:
            keys = [(t, mm, ww) for mm in self.M_t.get(t, []) for ww in self.W_t.get(t, [])]
            expr = gp.quicksum(self.p[(t, mm, ww)] * self.x[(t, mm, ww)] for (t, mm, ww) in keys)
            m.addConstr(self.C[t] == self.S[t] + expr, name=f"comp_time_{t}")

        # 3. Precedence & Time Lags: S_k >= C_i + L_{ik} for (i,k) in P
        for (i, k) in self.P:
            lag = self.L.get((i, k), 0.0)
            m.addConstr(self.S[k] >= self.C[i] + lag, name=f"lag_{i}_{k}")

        # 4. Job Windows: S_t >= r_j and C_j = max(C_t) <= d_j
        for j, tlist in self.job_tasks.items():
            rj = self.r.get(j, 0.0)
            dj = self.d.get(j, float('inf'))
            for t in tlist:
                m.addConstr(self.S[t] >= rj, name=f"release_{j}_{t}")
                # enforce each task completion <= deadline to ensure max <= d_j
                if dj != float('inf'):
                    m.addConstr(self.C[t] <= dj, name=f"deadline_{j}_{t}")

        # 5. Machine Sequence Consistency:
        # y_{i,k}^m <= sum_w x_{i,m,w} and y_{i,k}^m <= sum_w x_{k,m,w}
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for mm in self.machines:
                    # sum over workers for i on machine mm
                    xi_keys = [(i, mm, ww) for ww in self.W_t.get(i, []) if (i, mm, ww) in self.x]
                    xk_keys = [(k, mm, ww) for ww in self.W_t.get(k, []) if (k, mm, ww) in self.x]
                    if xi_keys:
                        m.addConstr(self.y[(i, k, mm)] <= gp.quicksum(self.x[k_] for k_ in xi_keys), name=f"y_le_xi_{i}_{k}_{mm}")
                    else:
                        # if i cannot be on mm then y must be 0
                        m.addConstr(self.y[(i, k, mm)] <= 0, name=f"y_zero_i_{i}_{k}_{mm}")
                    if xk_keys:
                        m.addConstr(self.y[(i, k, mm)] <= gp.quicksum(self.x[k_] for k_ in xk_keys), name=f"y_le_xk_{i}_{k}_{mm}")
                    else:
                        m.addConstr(self.y[(i, k, mm)] <= 0, name=f"y_zero_k_{i}_{k}_{mm}")

                    # y_{i,k}^m + y_{k,i}^m <= 1
                    m.addConstr(self.y[(i, k, mm)] + self.y[(k, i, mm)] <= 1, name=f"y_antisym_{i}_{k}_{mm}")

        # 6. Machine Disjunction with SDST (Big-M):
        # S_k >= C_i + s_{i,k,m} * y_{i,k}^m - V * (1 - y_{i,k}^m)
        # S_i >= C_k + s_{k,i,m} * y_{k,i}^m - V * (1 - y_{k,i}^m)
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for mm in self.machines:
                    y_ikm = self.y[(i, k, mm)]
                    s_ikm = self.s.get((i, k, mm), 0.0)
                    m.addConstr(self.S[k] >= self.C[i] + s_ikm * y_ikm - self.V * (1 - y_ikm), name=f"sdst1_{i}_{k}_{mm}")
                    # symmetric
                    y_kim = self.y[(k, i, mm)]
                    s_kim = self.s.get((k, i, mm), 0.0)
                    m.addConstr(self.S[i] >= self.C[k] + s_kim * y_kim - self.V * (1 - y_kim), name=f"sdst2_{i}_{k}_{mm}")

        # 7. Worker Sequence Consistency:
        # z_{i,k}^w <= sum_m x_{i,m,w} and z_{i,k}^w <= sum_m x_{k,m,w}
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for ww in self.workers:
                    xi_keys = [(i, mm, ww) for mm in self.M_t.get(i, []) if (i, mm, ww) in self.x]
                    xk_keys = [(k, mm, ww) for mm in self.M_t.get(k, []) if (k, mm, ww) in self.x]
                    if xi_keys:
                        m.addConstr(self.z[(i, k, ww)] <= gp.quicksum(self.x[k_] for k_ in xi_keys), name=f"z_le_xi_{i}_{k}_{ww}")
                    else:
                        m.addConstr(self.z[(i, k, ww)] <= 0, name=f"z_zero_i_{i}_{k}_{ww}")
                    if xk_keys:
                        m.addConstr(self.z[(i, k, ww)] <= gp.quicksum(self.x[k_] for k_ in xk_keys), name=f"z_le_xk_{i}_{k}_{ww}")
                    else:
                        m.addConstr(self.z[(i, k, ww)] <= 0, name=f"z_zero_k_{i}_{k}_{ww}")

                    # z_{i,k}^w + z_{k,i}^w <= 1
                    m.addConstr(self.z[(i, k, ww)] + self.z[(k, i, ww)] <= 1, name=f"z_antisym_{i}_{k}_{ww}")

        # 8. Worker Disjunction (Big-M):
        # S_k >= C_i - V * (1 - z_{i,k}^w)
        # S_i >= C_k - V * (1 - z_{k,i}^w)
        for i in self.tasks:
            for k in self.tasks:
                if i == k:
                    continue
                for ww in self.workers:
                    zijw = self.z[(i, k, ww)]
                    m.addConstr(self.S[k] >= self.C[i] - self.V * (1 - zijw), name=f"worker_disj1_{i}_{k}_{ww}")
                    zkjw = self.z[(k, i, ww)]
                    m.addConstr(self.S[i] >= self.C[k] - self.V * (1 - zkjw), name=f"worker_disj2_{i}_{k}_{ww}")

        # 9. Makespan: C_max >= C_t for all t
        for t in self.tasks:
            m.addConstr(self.C_max >= self.C[t], name=f"makespan_{t}")

        # Objective: minimize C_max
        m.setObjective(self.C_max, GRB.MINIMIZE)
        m.update()

    def optimize(self, time_limit: float = None):
        """Optimize the model.

        Optional: set a time limit in seconds.

        Returns a dict with keys:
          - status: textual solver status
          - gurobi_status: numeric status code
          - obj_val: objective value or None
        """
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


# End of file
