"""
Declarative CP model for Extended JSSP using OR-Tools CP-SAT (cp_model).

Class: JSSPCpModel
- __init__(data, cp_params=None): stores data and builds model
- build_model(): constructs variables, intervals and constraints
- optimize(time_limit_seconds=None): solves and returns results

Data dictionary expected keys:
  - tasks, job_tasks, machines, workers, P, L, M_t, W_t, p, s, V, release_dates, deadlines

Notes about implementation choices:
- Uses AddCircuit for machine sequencing with boolean arc variables.
- Uses AddNoOverlap for workers.

"""
from typing import Dict, List, Tuple, Any
from ortools.sat.python import cp_model


class JSSPCpModel:
    """CP-SAT model for the extended JSSP as specified.

    The data dictionary should follow the same conventions used by the MILP
    implementation. Durations, lags and deadlines must be integer or castable
    to int. The Big-M 'V' should be provided and used for the conditional
    inequalities linking task intervals and mode intervals.
    """

    def __init__(self, data: Dict[str, Any], cp_params: Dict[str, Any] = None):
        self.data = data
        self.params = cp_params or {}
        self.model = cp_model.CpModel()

        # will be filled during build
        self.horizon = None
        self.task_start = {}
        self.task_end = {}
        self.task_iv = {}
        self.mode_pres = {}  # (t,m,w) -> BoolVar
        self.mode_start = {}
        self.mode_end = {}
        self.mode_iv = {}
        self.job_start = {}
        self.job_end = {}
        self.job_iv = {}
        self.C_max = None

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

        if "job_tasks" in d:
            self.job_tasks: Dict[Any, List[Any]] = {j: list(ts) for j, ts in d["job_tasks"].items()}
            self.jobs = list(self.job_tasks.keys())
        else:
            raise KeyError("Data must provide 'job_tasks'")

        # precedence and lags
        self.P: List[Tuple[Any, Any]] = list(d.get("P", []))
        self.L: Dict[Tuple[Any, Any], int] = {tuple(k): int(v) for k, v in d.get("L", {}).items()} if d.get("L") else {}

        # release and deadlines
        self.r: Dict[Any, int] = {j: int(v) for j, v in d.get("release_dates", {}).items()} if d.get("release_dates") else {}
        self.d: Dict[Any, int] = {j: int(v) for j, v in d.get("deadlines", {}).items()} if d.get("deadlines") else {}

        # eligibility per task
        self.M_t: Dict[Any, List[Any]] = {t: list(ms) for t, ms in d["M_t"].items()}
        # worker->machines capability
        self.W_m: Dict[Any, List[Any]] = {w: list(ms) for w, ms in d["W_m"].items()}

        # processing times
        self.p: Dict[Tuple[Any, Any, Any], int] = {}
        for key, value in d["p"].items():
            self.p[tuple(key)] = int(value)

        # setups
        self.s: Dict[Tuple[Any, Any, Any], int] = {}
        if d.get("s"):
            for key, value in d["s"].items():
                self.s[tuple(key)] = int(value)

        # --- construct OM_t: feasible (machine,worker) modes per task ---
        self.OM_t: Dict[Any, List[Tuple[Any, Any]]] = {}
        for t in self.tasks:
            om_list = []
            for mm in self.M_t.get(t, []):
                for ww in self.workers:
                    if mm in self.W_m.get(ww, []) and (t, mm, ww) in self.p:
                        om_list.append((mm, ww))
            self.OM_t[t] = om_list
            if not om_list:
                raise ValueError(f"Task {t} has no feasible operation modes OM_t")

        # compute a safe horizon (upper bound on time) for variable domains
        max_deadline = max(self.d.values()) if self.d else 0
        max_release = max(self.r.values()) if self.r else 0
        sum_max_p = 0
        for t in self.tasks:
            max_p_t = 0
            for (mm, ww) in self.OM_t.get(t, []):
                max_p_t = max(max_p_t, self.p.get((t, mm, ww), 0))
            sum_max_p += max_p_t
        # max setup
        max_s = max(self.s.values()) if self.s else 0
        # horizon estimate
        self.horizon = max(max_deadline, max_release + sum_max_p + max_s * len(self.tasks) )

    def build_model(self):
        model = self.model

        # Create job interval vars I_j
        for j in self.jobs:
            rj = self.r.get(j, 0)
            dj = self.d.get(j, self.horizon)
            start_j = model.NewIntVar(rj, dj, f"start_job_{j}")
            end_j = model.NewIntVar(rj, dj, f"end_job_{j}")
            length_j = model.NewIntVar(0, dj - rj, f"len_job_{j}")
            iv_j = model.NewIntervalVar(start_j, length_j, end_j, f"job_iv_{j}")
            self.job_start[j] = start_j
            self.job_end[j] = end_j
            self.job_iv[j] = iv_j

        # Create mode optional intervals O_{t,m,w} and task intervals I_t
        for t in self.tasks:
            s_t = model.NewIntVar(0, self.horizon, f"start_task_{t}")
            e_t = model.NewIntVar(0, self.horizon, f"end_task_{t}")
            len_t = model.NewIntVar(0, self.horizon, f"len_task_{t}")
            iv_t = model.NewIntervalVar(s_t, len_t, e_t, f"task_iv_{t}")
            self.task_start[t] = s_t
            self.task_end[t] = e_t
            self.task_iv[t] = iv_t

            mode_pres_list = []
            # iterate only over feasible OM_t modes
            for (mm, ww) in self.OM_t.get(t, []):
                dur = self.p.get((t, mm, ww), None)
                if dur is None:
                    continue
                pres = model.NewBoolVar(f"pres_{t}_{mm}_{ww}")
                mo_start = model.NewIntVar(0, self.horizon, f"start_mode_{t}_{mm}_{ww}")
                mo_end = model.NewIntVar(0, self.horizon, f"end_mode_{t}_{mm}_{ww}")
                dur_var = dur
                iv_mode = model.NewOptionalIntervalVar(mo_start, dur_var, mo_end, pres, f"mode_iv_{t}_{mm}_{ww}")

                self.mode_pres[(t, mm, ww)] = pres
                self.mode_start[(t, mm, ww)] = mo_start
                self.mode_end[(t, mm, ww)] = mo_end
                self.mode_iv[(t, mm, ww)] = iv_mode
                mode_pres_list.append(pres)

            if not mode_pres_list:
                raise ValueError(f"Task {t} has no eligible (machine,worker) modes")
            model.AddExactlyOne(mode_pres_list)

            # Synchronize task interval with selected mode; iterating mode_pres keys is enough
            for (tt, mm, ww), pres in list(self.mode_pres.items()):
                if tt != t:
                    continue
                ms = self.mode_start[(tt, mm, ww)]
                me = self.mode_end[(tt, mm, ww)]
                dur_var = self.p.get((tt, mm, ww))
                model.Add(self.task_start[t] == ms).OnlyEnforceIf(pres)
                model.Add(self.task_end[t] == me).OnlyEnforceIf(pres)
                model.Add(len_t == dur_var).OnlyEnforceIf(pres)

        # 3. Precedence & Time Lags: Start(I_k) >= End(I_i) + L_{ik}
        for (i, k) in self.P:
            lag = self.L.get((i, k), 0)
            model.Add(self.task_start[k] >= self.task_end[i] + lag)

        # 4. Job Windows
        for j, tlist in self.job_tasks.items():
            rj = self.r.get(j, 0)
            dj = self.d.get(j, self.horizon)
            for t in tlist:
                model.Add(self.job_start[j] <= self.task_start[t])
                model.Add(self.job_end[j] >= self.task_end[t])
            model.Add(self.job_start[j] >= rj)
            model.Add(self.job_end[j] <= dj)

        # 5. Worker Capacity: NoOverlap over optional intervals assigned to worker w
        for ww in self.workers:
            ivs = []
            for (t, mm, ww2), iv in self.mode_iv.items():
                if ww2 != ww:
                    continue
                # mode_iv keys are only feasible OM_t modes so filtering by worker is sufficient
                ivs.append(iv)
            model.AddNoOverlap(ivs)

        # 6. Machine Sequencing & SDST using AddCircuit
        for mm in self.machines:
            node_id = 1
            mode_to_node = {}
            node_to_mode = {0: None}
            for (t, m2, w) in self.mode_iv.keys():
                if m2 != mm:
                    continue
                mode_to_node[(t, m2, w)] = node_id
                node_to_mode[node_id] = (t, m2, w)
                node_id += 1
            num_nodes = node_id
            arcs = []
            arc_var = {}
            for u in range(num_nodes):
                for v in range(num_nodes):
                    b = model.NewBoolVar(f"arc_m_{mm}_{u}_{v}")
                    arcs.append((u, v, b))
                    arc_var[(u, v)] = b
                    if u == v:
                        if u != 0:
                            mode_u = node_to_mode[u]
                            pres_u = self.mode_pres[mode_u]
                            model.Add(pres_u == 0).OnlyEnforceIf(b)
                            model.Add(b == 1).OnlyEnforceIf(pres_u.Not())
                    else:
                        if u != 0 and v != 0:
                            mode_u = node_to_mode[u]
                            mode_v = node_to_mode[v]
                            pres_u = self.mode_pres[mode_u]
                            pres_v = self.mode_pres[mode_v]
                            model.AddImplication(b, pres_u)
                            model.AddImplication(b, pres_v)
                            s_uv = self.s.get((mode_u[0], mode_v[0], mm), 0)
                            start_v = self.mode_start[mode_v]
                            end_u = self.mode_end[mode_u]
                            model.Add(start_v >= end_u + s_uv).OnlyEnforceIf(b)
                        elif u != 0 and v == 0:
                            mode_u = node_to_mode[u]
                            model.AddImplication(b, self.mode_pres[mode_u])
                        elif u == 0 and v != 0:
                            mode_v = node_to_mode[v]
                            model.AddImplication(b, self.mode_pres[mode_v])
            model.AddCircuit(arcs)

        # 7. Objective: C_max >= End(I_j) for all j; minimize C_max
        self.C_max = model.NewIntVar(0, self.horizon, "C_max")
        for j in self.jobs:
            model.Add(self.C_max >= self.job_end[j])
        model.Minimize(self.C_max)

    def optimize(self, time_limit_seconds: int = None):
        solver = cp_model.CpSolver()
        # apply params
        if time_limit_seconds is not None:
            solver.parameters.max_time_in_seconds = float(time_limit_seconds)
        # apply user params
        for k, v in self.params.items():
            try:
                setattr(solver.parameters, k, v)
            except Exception:
                # ignore unknown params
                pass

        status = solver.Solve(self.model)
        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
        }

        res_status = status_map.get(status, f"STATUS_{status}")
        obj = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj = solver.Value(self.C_max)

        # Collect solution if available
        solution = {}
        if obj is not None:
            # tasks start/end
            solution["tasks"] = {t: {"start": solver.Value(self.task_start[t]), "end": solver.Value(self.task_end[t])} for t in self.tasks}
            # modes selected
            selected = []
            for (t, m, w), pres in self.mode_pres.items():
                if solver.Value(pres):
                    selected.append((t, m, w))
            solution["selected_modes"] = selected
            # jobs
            solution["jobs"] = {j: {"start": solver.Value(self.job_start[j]), "end": solver.Value(self.job_end[j])} for j in self.jobs}

        return {"status": res_status, "obj_val": obj, "solution": solution}


