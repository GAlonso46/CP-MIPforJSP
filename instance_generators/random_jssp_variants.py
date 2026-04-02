"""
Instance variant generators for extended JSSP.

Provides the following functions:
- generate_fjssp_variant
- generate_timelags_variant
- generate_release_dates_variant

All functions return a new data dictionary (deep copy) so the original input
is not modified.
"""
from typing import Dict, Any, List, Tuple
import random
import copy


def generate_fjssp_variant(data: Dict[str, Any], min_m: int = 1, max_m: int = 3, seed: int = None) -> Dict[str, Any]:
    """Generate a Flexible JSSP variant by assigning a random subset of machines
    to each task and creating consistent processing times for each (t,m,w).

    - min_m, max_m: minimum/maximum number of eligible machines per task.
    - The function preserves the set of workers (W_t) for tasks. For each new
      (task, machine, worker) combination a processing time is sampled around
      the original task's typical processing time.
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    machines = list(out.get("machines", []))
    workers = list(out.get("workers", []))

    # Ensure raw p exists; convert keys to tuples
    orig_p = {tuple(k): float(v) for k, v in out.get("p", {}).items()}

    # compute a baseline processing time per task from original p if available
    baseline = {}
    for t in out.get("tasks", []):
        vals = [v for (tt, mm, ww), v in orig_p.items() if tt == t]
        if vals:
            baseline[t] = max(1.0, sum(vals) / len(vals))
        else:
            baseline[t] = 1.0

    # build new M_t and p
    new_M_t: Dict[Any, List[Any]] = {}
    new_p: Dict[Tuple[Any, Any, Any], float] = {}

    for t in out.get("tasks", []):
        # choose k machines between min_m and max_m (clipped to available)
        k = rnd.randint(min_m, max(min_m, min(max_m, len(machines))))
        chosen = rnd.sample(machines, k)
        new_M_t[t] = chosen

        # for each chosen machine and each eligible worker, assign a processing time
        b = baseline.get(t, 1.0)
        # define sampling window relative to baseline
        low = max(1, int(max(1, b * 0.5)))
        high = max(low, int(b * 1.5) + 1)
        for m in chosen:
            # use provided W_t if exists, else all workers
            wlist = out.get("W_t", {}).get(t, workers)
            for w in wlist:
                pt = float(rnd.randint(low, high))
                new_p[(t, m, w)] = pt

    out["M_t"] = new_M_t
    out["p"] = {k: v for k, v in new_p.items()}
    # keep W_t unchanged
    return out


def generate_timelags_variant(data: Dict[str, Any], density: float = 0.3, min_lag: int = 1, max_lag: int = 10, seed: int = None) -> Dict[str, Any]:
    """Add time lags between consecutive tasks of the same job.

    For each consecutive pair (t_i, t_{i+1}) in a job, with probability
    `density` add a lag uniformly sampled between min_lag and max_lag.
    If no lag is created (zero total), force one random consecutive pair to
    have a lag so the variant is active.
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    L = {}
    tasks_with_pairs: List[Tuple[Any, Any]] = []
    for j, tlist in out.get("job_tasks", {}).items():
        for idx in range(len(tlist) - 1):
            i = tlist[idx]
            k = tlist[idx + 1]
            tasks_with_pairs.append((i, k))
            if rnd.random() <= density:
                L[(i, k)] = rnd.randint(min_lag, max_lag)

    # ensure at least one lag exists
    if len(L) == 0 and tasks_with_pairs:
        pair = rnd.choice(tasks_with_pairs)
        L[pair] = rnd.randint(min_lag, max_lag)

    out["L"] = {k: v for k, v in L.items()}
    return out


def generate_release_dates_variant(base_data: Dict[str, Any], job_prob: float = 0.4, max_r_ratio: float = 0.5, seed: int = None) -> Dict[str, Any]:
    """Generate release dates for a subset of jobs.

    - Compute a simple makespan upper bound as the sum of per-task maximum
      processing times (across modes). For each job, with probability
      job_prob, assign a release date uniformly in [0, makespan * max_r_ratio].
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(base_data)

    # estimate makespan upper bound (sum of max processing times per task)
    sum_max_p = 0
    for t in out.get("tasks", []):
        max_p_t = 0
        # if p uses tuple keys, iterate
        for (tt, mm, ww), val in out.get("p", {}).items():
            if tt == t:
                max_p_t = max(max_p_t, float(val))
        # fallback to 1 if nothing found
        sum_max_p += max(1.0, max_p_t)

    bound = float(sum_max_p)
    release_dates: Dict[Any, int] = {}
    for j in out.get("job_tasks", {}).keys():
        if rnd.random() <= job_prob:
            r = rnd.uniform(0, bound * float(max_r_ratio))
            release_dates[j] = int(round(r))

    out["release_dates"] = release_dates
    return out


def generate_sdst_uniform_variant(
    data: Dict[str, Any],
    alpha: float = 0.5,
    seed: int = None
) -> Dict[str, Any]:
    """Generate uniform sequence-dependent setup times (SDST).

    Returns a deep copy of the input data with key "s" added, mapping
    (task_i, task_j, machine) -> int setup time.

    Parameters:
    - data: input instance dictionary (not modified).
    - alpha: fraction of average processing time used as upper bound.
    - seed: seed for random generator.
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    # Compute average processing time over all p entries
    total = 0.0
    count = 0
    for _, val in out.get("p", {}).items():
        try:
            total += float(val)
            count += 1
        except Exception:
            continue
    avg_p = float(total / count) if count > 0 else 1.0

    # Integer upper bound for setup sampling
    max_setup = max(0, int(alpha * avg_p))

    tasks = list(out.get("tasks", []))
    machines = list(out.get("machines", []))

    s: Dict[Tuple[Any, Any, Any], int] = {}

    for m in machines:
        for i in tasks:
            for j in tasks:
                if i == j:
                    s[(i, j, m)] = 0
                else:
                    if max_setup <= 0:
                        s[(i, j, m)] = 0
                    else:
                        s[(i, j, m)] = int(rnd.randint(0, max_setup))

    out["s"] = s
    return out


def generate_sdst_mixed_variant(
    data: Dict[str, Any],
    alpha: float = 0.5,
    lambda_weight: float = 0.7,
    K: int = 3,
    beta_intra: float = 0.2,
    beta_inter_min: float = 0.5,
    beta_inter_max: float = 1.0,
    seed: int = None
) -> Dict[str, Any]:
    """Generate mixed SDST combining a uniform component and a cluster-based component.

    Returns a deep copy of the input data with key "s" added, mapping
    (task_i, task_j, machine) -> int setup time.

    Parameters:
    - alpha: fraction for uniform component bound.
    - lambda_weight: weight of uniform component in final value.
    - K: number of clusters.
    - beta_intra: fraction for intra-cluster component bound.
    - beta_inter_min/beta_inter_max: fractions for inter-cluster range.
    - seed: seed for random generator.
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    # Compute average processing time
    total = 0.0
    count = 0
    for _, val in out.get("p", {}).items():
        try:
            total += float(val)
            count += 1
        except Exception:
            continue
    avg_p = float(total / count) if count > 0 else 1.0

    # Integer bounds
    max_u = max(0, int(alpha * avg_p))
    max_intra = max(0, int(beta_intra * avg_p))
    inter_low = max(0, int(beta_inter_min * avg_p))
    inter_high = max(inter_low, int(beta_inter_max * avg_p))

    tasks = list(out.get("tasks", []))
    machines = list(out.get("machines", []))

    K_eff = max(1, int(K))

    # Assign clusters
    clusters: Dict[Any, int] = {}
    for t in tasks:
        clusters[t] = rnd.randrange(K_eff)

    s: Dict[Tuple[Any, Any, Any], int] = {}

    for m in machines:
        for i in tasks:
            for j in tasks:
                if i == j:
                    s[(i, j, m)] = 0
                    continue

                u_ij = rnd.randint(0, max_u) if max_u > 0 else 0

                if clusters.get(i) == clusters.get(j):
                    c_ij = rnd.randint(0, max_intra) if max_intra > 0 else 0
                else:
                    low = inter_low
                    high = inter_high
                    if high < low:
                        high = low
                    c_ij = rnd.randint(low, high) if high > 0 else 0

                # Weighted integer combination
                s_val = int(lambda_weight * u_ij + (1.0 - lambda_weight) * c_ij)
                s[(i, j, m)] = s_val

    out["s"] = s
    return out