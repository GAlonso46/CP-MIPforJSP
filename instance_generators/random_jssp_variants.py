"""
Instance variant generators for extended JSSP.

Provides the following functions:
- generate_fjssp_variant

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


