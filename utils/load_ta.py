"""
Parser to read a TA-format JSSP instance file and convert it into the data
structure expected by the project's models.

The returned dictionary contains keys compatible with `JSSPMilpModel` and
`JSSPCpModel`:
- machines: list of machine ids (ints)
- workers: list of worker ids (default: [0])
- tasks: list of task ids (we use tuples (job_id, op_idx))
- job_tasks: mapping job_id -> list of task ids (in operation order)
- M_t: mapping task -> list of eligible machines (single machine from TA file)
- W_m: mapping worker -> list of machines the worker can operate (default: {0: machines})
- p: mapping (task, machine, worker) -> processing_time (int)
- P: list of precedence arcs (task_prev, task_next)

This parser ignores lines that begin with '#' (comments).
"""
from typing import Dict, List, Tuple, Any
from pathlib import Path


def parse_ta_file(path: str) -> Dict[str, Any]:
    """Parse a TA-format file and return a data dictionary suitable for the models.

    Lines starting with '#' are ignored (comments). The parser expects the first
    non-comment non-empty line to contain two integers: num_jobs num_machines.
    Each following non-comment job line contains num_machines pairs: machine proc_time
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")

    text = p.read_text(encoding="utf-8")
    # split into non-empty, non-comment lines
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip() != "" and not ln.strip().startswith("#")]
    if len(raw_lines) == 0:
        raise ValueError("Empty or comment-only TA file")

    # header: first non-comment line
    header_tokens = raw_lines[0].split()
    if len(header_tokens) < 2:
        raise ValueError("Header must contain num_jobs and num_machines")
    try:
        num_jobs = int(header_tokens[0])
        num_machines = int(header_tokens[1])
    except Exception as exc:
        raise ValueError("Header tokens must be integers: num_jobs num_machines") from exc

    # gather remaining tokens from subsequent non-comment lines
    remaining_tokens: List[str] = []
    for ln in raw_lines[1:]:
        toks = ln.split()
        remaining_tokens.extend(toks)

    expected_tokens = num_jobs * num_machines * 2
    if len(remaining_tokens) < expected_tokens:
        raise ValueError(f"Not enough tokens for declared jobs/machines: expected {expected_tokens}, got {len(remaining_tokens)}")

    # parse tokens sequentially per job, each job has num_machines pairs
    idx = 0
    jobs_ops: List[List[Tuple[int, int]]] = []  # for each job a list of (machine, proc)
    for j in range(num_jobs):
        ops = []
        for _ in range(num_machines):
            try:
                machine = int(remaining_tokens[idx])
                proc = int(remaining_tokens[idx + 1])
            except Exception as exc:
                raise ValueError("Malformed machine/processing_time pair") from exc
            ops.append((machine, proc))
            idx += 2
        jobs_ops.append(ops)

    # Build model-compatible data structure
    machines = sorted({m for job in jobs_ops for (m, _) in job})
    # fallback: if file used different ids, ensure canonical range
    if len(machines) != num_machines:
        machines = list(range(num_machines))

    # single worker default (parser cannot infer workers from TA format)
    workers = []
    # worker->machines capability: default worker can operate all machines
    W_m: Dict[Any, List[Any]] = {}

    tasks = []
    job_tasks: Dict[Any, List[Any]] = {}
    M_t: Dict[Any, List[Any]] = {}
    p_dict: Dict[Tuple[Any, Any, Any], int] = {}

    for j_idx, ops in enumerate(jobs_ops):
        job_task_list: List[Any] = []
        for op_idx, (machine, proc) in enumerate(ops):
            task_id = (j_idx, op_idx)
            tasks.append(task_id)
            job_task_list.append(task_id)
            M_t[task_id] = [machine]
            # only create p entries for feasible triples: here worker 0 can do all machines
            p_dict[(task_id, machine, 0)] = int(proc)
        job_tasks[j_idx] = job_task_list

    # precedence arcs: within each job, op_k -> op_{k+1}
    P: List[Tuple[Any, Any]] = []
    for j_idx, tlist in job_tasks.items():
        for k in range(len(tlist) - 1):
            P.append((tlist[k], tlist[k + 1]))

    data = {
        "machines": machines,
        "workers": workers,
        "W_m": W_m,
        "tasks": tasks,
        "job_tasks": job_tasks,
        "M_t": M_t,
        "p": p_dict,
        "P": P,
    }

    return data
