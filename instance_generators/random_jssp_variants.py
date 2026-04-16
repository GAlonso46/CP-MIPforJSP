"""
Instance variant generators for extended JSSP.

Provides the following functions:
- generate_fjssp_variant
- generate_timelags_variant
- generate_release_dates_variant
- generate_sdst_uniform_variant
- generate_sdst_mixed_variant
- generate_deadlines_variant
- generate_dual_resources_variant

All functions return a new data dictionary (deep copy) so the original input
is not modified.
"""
from typing import Dict, Any, List, Tuple
import random
import copy
import math


def generate_fjssp_variant(
    data: Dict[str, Any],
    min_m: int = 1,
    max_m_fraction: float = 0.5,
    variation: float = 0.2,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate a Flexible JSSP variant.

    Key features:
    - Each task is assigned a random subset of eligible machines.
    - The maximum number of machines is defined as a fraction of total machines.
    - Processing times are generated coherently per task:
        - Each task has a baseline processing time.
        - Each machine gets a value in a proportional neighborhood of the baseline.
        - The same processing time is used for all compatible workers on that machine.

    Parameters
    ----------
    data : Dict[str, Any]
        Base JSSP instance.
    min_m : int
        Minimum number of machines per task.
    max_m_fraction : float
        Fraction of total machines used to compute maximum machines per task.
    variation : float
        Relative variation around the baseline processing time (e.g., 0.2 = ±20%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        New instance with updated M_t and p.
    """

    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    machines: List[Any] = list(out.get("machines", []))
    workers: List[Any] = list(out.get("workers", []))

    # Convert original processing times to consistent format
    orig_p = {tuple(k): float(v) for k, v in out.get("p", {}).items()}

    # --- Compute baseline processing time per task ---
    baseline: Dict[Any, float] = {}
    for t in out.get("tasks", []):
        vals = [v for (tt, mm, ww), v in orig_p.items() if tt == t]
        if vals:
            baseline[t] = max(1.0, sum(vals) / len(vals))
        else:
            baseline[t] = 1.0

    # --- Compute max machines from fraction ---
    num_machines = len(machines)
    max_m = max(min_m, int(max_m_fraction * num_machines))
    max_m = min(max_m, num_machines)

    # --- Build machine -> workers mapping ---
    W_m = out.get("W_m", {})
    machine_to_workers: Dict[Any, List[Any]] = {m: [] for m in machines}
    for w, m_list in W_m.items():
        for m in m_list:
            if m in machine_to_workers:
                machine_to_workers[m].append(w)

    new_M_t: Dict[Any, List[Any]] = {}
    new_p: Dict[Tuple[Any, Any, Any], int] = {}

    # --- Main generation ---
    for t in out.get("tasks", []):
        # Select number of machines
        k = rnd.randint(min_m, max_m)
        chosen_machines = rnd.sample(machines, k)
        new_M_t[t] = chosen_machines

        b = baseline.get(t, 1.0)

        # Compute proportional neighborhood
        low = max(1, int((1.0 - variation) * b))
        high = max(low + 1, int((1.0 + variation) * b))

        for m in chosen_machines:
            # Generate one processing time per (task, machine)
            pt_m = rnd.randint(low, high)

            # Assign same value to all compatible workers
            for w in machine_to_workers.get(m, []):
                new_p[(t, m, w)] = pt_m

    out["M_t"] = new_M_t
    out["p"] = new_p

    out["metadata"] = {
        "seed": seed,
        "generator_version": "1.0",
        "variant": "FJSSP",
        "creation_date": __import__("datetime").date.today().isoformat(),
    }

    return out


def generate_timelags_variant(
    data: Dict[str, Any],
    density: float = 0.3,
    min_lag: int = 1,
    max_lag_factor: float = 0.5,
    seed: int = None
) -> Dict[str, Any]:
    """
    Add time lags between consecutive tasks of the same job.

    For each consecutive pair (t_i, t_{i+1}) in a job, with probability
    `density` a lag is added. The lag is uniformly sampled between
    `min_lag` and a dynamically computed maximum lag based on the
    average processing time of the instance.

    The maximum lag is defined as:
        max_lag = max_lag_factor * avg_processing_time

    If no lag is created (i.e., zero total), one random consecutive pair
    is forced to have a lag to ensure the variant is active.
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    # --- Compute average processing time ---
    p_values = list(out.get("p", {}).values())
    if not p_values:
        raise ValueError("Processing times 'p' cannot be empty.")

    avg_p = sum(p_values) / len(p_values)

    # Compute dynamic max_lag (ensure integer and >= min_lag)
    max_lag = max(min_lag, int(max_lag_factor * avg_p))

    L = {}
    tasks_with_pairs: List[Tuple[Any, Any]] = []

    # --- Generate time lags ---
    for j, tlist in out.get("job_tasks", {}).items():
        for idx in range(len(tlist) - 1):
            i = tlist[idx]
            k = tlist[idx + 1]
            tasks_with_pairs.append((i, k))

            if rnd.random() <= density:
                L[(i, k)] = rnd.randint(min_lag, max_lag)

    # --- Ensure at least one lag exists ---
    if len(L) == 0 and tasks_with_pairs:
        pair = rnd.choice(tasks_with_pairs)
        L[pair] = rnd.randint(min_lag, max_lag)

    out["L"] = {k: v for k, v in L.items()}
    out["metadata"] = {
        "seed": seed,
        "generator_version": "1.0",
        "variant": "TJSSP",
        "creation_date": __import__("datetime").date.today().isoformat(),
    }

    return out


def generate_release_dates_variant(
    data: Dict[str, Any],
    job_prob: float = 0.4,
    max_r_ratio: float = 0.5,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate release dates for a subset of jobs based on estimated average machine workload.

    The upper bound for release dates is computed as the total average processing
    time per task divided by the number of machines. For each job, with probability
    `job_prob`, a release date is sampled uniformly in [0, bound * max_r_ratio].
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    p_dict = out.get("p", {})
    if not p_dict:
        raise ValueError("Processing times 'p' cannot be empty.")

    # --- Identify machines and group processing times by task ---
    machines = set()
    task_p: Dict[Any, list] = {}

    for (t, m, w), val in p_dict.items():
        machines.add(m)
        if t not in task_p:
            task_p[t] = []
        task_p[t].append(int(val))

    num_machines = len(machines) if machines else 1

    # --- Compute sum of average processing time per task ---
    sum_avg_p = 0
    for t, p_list in task_p.items():
        avg_t = sum(p_list) / len(p_list)
        sum_avg_p += avg_t

    # --- Estimated average workload per machine ---
    bound = sum_avg_p / num_machines

    # --- Generate release dates ---
    release_dates: Dict[Any, int] = {}

    for j in out.get("job_tasks", {}).keys():
        if rnd.random() <= job_prob:
            r = rnd.randint(0, int(bound * max_r_ratio))
            release_dates[j] = r

    out["release_dates"] = release_dates
    out["metadata"] = {
        "seed": seed,
        "generator_version": "1.0",
        "variant": "RJSSP",
        "creation_date": __import__("datetime").date.today().isoformat(),
    }

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
    out["metadata"] = {
        "seed": seed,
        "generator_version": "1.0",
        "variant": "SDSTJSSP",
        "creation_date": __import__("datetime").date.today().isoformat(),
    }
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
    out["metadata"] = {
        "seed": seed,
        "generator_version": "2.0",
        "variant": "SDSTJSSP",
        "creation_date": __import__("datetime").date.today().isoformat(),
    }
    return out


def generate_deadlines_variant(
    data: Dict[str, Any],
    deadline_density: float = 0.3,
    gamma_min: float = 1.3,
    gamma_max: float = 1.6,
    noise_ratio: float = 0.2,
    seed: int = None
) -> Dict[str, Any]:
    """Generate integer deadlines for a subset of jobs.

    Parameters:
    - data: input instance dictionary (not modified).
    - deadline_density: fraction of jobs to assign deadlines (floor applied).
    - gamma_min/gamma_max: range for multiplicative factor on job duration.
    - noise_ratio: fraction of L_j used as upper bound for additive noise.
    - seed: random seed for reproducibility.

    Behavior:
    - Select exactly floor(deadline_density * num_jobs) jobs uniformly at random
      (without replacement).
    - Compute job duration L_j as the sum of per-task processing times. For each
      task, any available (machine, worker) processing time from `p` is used
      (the minimum found is selected; fallback to 1 if none).
    - For each selected job j, sample gamma_j ~ Uniform(gamma_min, gamma_max)
      and noise ~ Uniform(0, noise_ratio * L_j). Deadline d_j = r_j +
      gamma_j * L_j + noise, then rounded up to an integer and ensured to be
      at least r_j + L_j.

    Returns a deep copy of `data` with key "deadlines" mapping job_id -> int.
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    job_tasks: Dict[Any, List[Any]] = out.get("job_tasks", {})
    jobs = list(job_tasks.keys())
    num_jobs = len(jobs)

    # number of deadlines to assign
    k = int(math.floor(deadline_density * num_jobs)) if num_jobs > 0 else 0

    selected_jobs: List[Any] = rnd.sample(jobs, k) if k > 0 else []

    # prepare p with tuple keys for easy lookup
    orig_p: Dict[Tuple[Any, Any, Any], float] = {tuple(k): float(v) for k, v in out.get("p", {}).items()}

    deadlines: Dict[Any, int] = {}

    for j in selected_jobs:
        tasks = job_tasks.get(j, [])
        # compute L_j as sum of a chosen processing time per task
        L_j = 0.0
        for t in tasks:
            vals = [v for (tt, mm, ww), v in orig_p.items() if tt == t]
            if vals:
                # choose the minimum available processing time for stability
                L_j += float(min(vals))
            else:
                L_j += 1.0

        # release date for job (default 0)
        r_j = int(out.get("release_dates", {}).get(j, 0))

        # sample gamma and noise
        gamma_j = rnd.uniform(float(gamma_min), float(gamma_max))
        noise = rnd.uniform(0.0, float(noise_ratio) * L_j) if L_j > 0 else 0.0

        d = float(r_j) + gamma_j * L_j + noise
        min_allowed = float(r_j) + L_j
        if d < min_allowed:
            d = min_allowed

        # round up to integer to ensure deadline >= min_allowed
        d_int = int(math.ceil(d))
        deadlines[j] = d_int

    out["deadlines"] = deadlines
    out["metadata"] = {
        "seed": seed,
        "generator_version": "1.0",
        "variant": "DJSSP",
        "creation_date": __import__("datetime").date.today().isoformat(),
    }
    return out


def generate_dual_resources_variant(
    data: Dict[str, Any],
    rho_workers: float = 1.0,
    compatibility_prob: float = 0.3,
    delta: float = 0.2,
    seed: int = None
) -> Dict[str, Any]:
    """Add dual resources (workers) and compatible processing times.

    Parameters:
    - data: input instance dictionary (not modified).
    - rho_workers: multiplier for number of workers relative to machines
      (num_workers = floor(rho_workers * len(machines))).
    - compatibility_prob: probability a worker is compatible with a machine
      during the random compatibility step.
    - delta: relative noise range for processing times (epsilon in [-delta,delta]).
    - seed: random seed for reproducibility.

    Returns a deep copy of `data` with updated keys:
    - "workers": list of worker ids (ints)
    - "W_m": mapping worker -> list of machines the worker can operate
    - "p": mapping (task, machine, worker) -> int processing time

    Notes:
    - If no base processing time is found for a (task,machine) pair, a
      default of 1 is used.
    - All generated processing times are integers >= 1.
    """
    rnd = random.Random(seed)
    out = copy.deepcopy(data)

    machines = list(out.get("machines", []))
    tasks = list(out.get("tasks", []))

    num_machines = len(machines)
    num_workers = int(float(rho_workers) * float(num_machines)) if num_machines > 0 else 0

    # generate worker ids 0..num_workers-1
    workers = list(range(num_workers))

    # initialize worker -> set(machine) mapping
    W_m_worker: Dict[int, set] = {w: set() for w in workers}

    # Step 1: coverage - ensure every machine has at least one worker
    if workers:
        for m in machines:
            w = rnd.choice(workers)
            W_m_worker[w].add(m)

        # Step 2: random compatibility for remaining worker-machine pairs
        for w in workers:
            assigned = W_m_worker[w]
            for m in machines:
                if m in assigned:
                    continue
                if rnd.random() <= float(compatibility_prob):
                    assigned.add(m)

    # convert sets to sorted lists for determinism
    W_m_out: Dict[Any, List[Any]] = {w: sorted(list(ms)) for w, ms in W_m_worker.items()}

    # Prepare base processing times per (task,machine)
    orig_p: Dict[Tuple[Any, Any, Any], float] = {tuple(k): float(v) for k, v in out.get("p", {}).items()}
    base_p: Dict[Tuple[Any, Any], float] = {}
    for (t, m, w), val in orig_p.items():
        key = (t, m)
        if key in base_p:
            # keep minimum found as stable baseline
            base_p[key] = min(base_p[key], float(val))
        else:
            base_p[key] = float(val)

    # Build new p for all compatible (task,machine,worker)
    new_p: Dict[Tuple[Any, Any, Any], int] = {}
    for t in tasks:
        for m in machines:
            key = (t, m)
            p_base = float(base_p.get(key, 1.0))
            if not workers:
                # no workers: preserve any existing worker-agnostic entries if present
                # try to keep original entries unchanged
                vals = [v for (tt, mm, ww), v in orig_p.items() if tt == t and mm == m]
                if vals:
                    # reinsert existing tuples (preserve worker identifier if present)
                    for (tt, mm, ww), v in orig_p.items():
                        if tt == t and mm == m:
                            new_p[(t, m, ww)] = max(1, int(float(v)))
                else:
                    # no original entry, nothing to add
                    continue
            else:
                for w in workers:
                    if m not in W_m_worker.get(w, set()):
                        continue
                    # sample epsilon in [-delta, delta]
                    eps = rnd.uniform(-float(delta), float(delta))
                    val = float(p_base) * (1.0 + eps)
                    # convert to integer and ensure at least 1
                    p_int = max(1, int(val))
                    new_p[(t, m, w)] = p_int

    # update out
    out["workers"] = workers
    out["W_m"] = W_m_out
    # represent p keys as tuples (they already are)
    out["p"] = {k: v for k, v in new_p.items()}

    out["metadata"] = {
        "seed": seed,
        "generator_version": "1.0",
        "variant": "DRCJSSP",
        "creation_date": __import__("datetime").date.today().isoformat(),
    }
    return out