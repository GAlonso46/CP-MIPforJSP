"""
Parser to read a TA-format JSSP instance file and convert it into the data
structure expected by the project's models.

The returned dictionary contains at least the keys required by `jssp_milp.JSSPMilpModel`:
- machines: list of machine ids (ints)
- workers: list of worker ids (default: [0])
- tasks: list of task ids (we use tuples (job_id, op_idx))
- job_tasks: mapping job_id -> list of task ids (in operation order)
- M_t: mapping task -> list of eligible machines (single machine from TA file)
- W_t: mapping task -> list of eligible workers (default [0])
- p: mapping (task, machine, worker) -> processing_time
- P: list of precedence arcs (task_prev, task_next)
- V: big-M constant (set to sum of processing times + 1)

"""
from typing import Dict, List, Tuple, Any
from pathlib import Path


def parse_ta_file(path: str) -> Dict[str, Any]:
    """Parse a TA-format file and return a data dictionary suitable for the models.

    Args:
        path: path to the TA-format file (string or Path).

    Returns:
        A dictionary with keys compatible with `JSSPMilpModel`.

    Raises:
        ValueError: if the file cannot be parsed or the content is malformed.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")

    text = p.read_text(encoding="utf-8")
    # split into non-empty lines
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if len(raw_lines) == 0:
        raise ValueError("Empty TA file")

    # header: first non-empty line
    header_tokens = raw_lines[0].split()
    if len(header_tokens) < 2:
        raise ValueError("Header must contain num_jobs and num_machines")
    try:
        num_jobs = int(header_tokens[0])
        num_machines = int(header_tokens[1])
    except Exception as exc:
        raise ValueError("Header tokens must be integers: num_jobs num_machines") from exc

    # gather remaining tokens from subsequent lines (support arbitrary spacing)
    remaining_tokens: List[str] = []
    for ln in raw_lines[1:]:
        # each job line contains pairs machine processing_time repeated num_machines times
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
    # ensure machine ids are 0..num_machines-1 or at least consistent
    # we'll trust file machine ids but also provide a canonical list 0..num_machines-1 if not present
    if len(machines) != num_machines:
        # fallback: use 0..num_machines-1
        machines = list(range(num_machines))

    workers = [0]

    tasks = []
    job_tasks = {}
    M_t = {}
    W_t = {}
    p_dict = {}
    total_proc = 0

    for j_idx, ops in enumerate(jobs_ops):
        job_task_list = []
        for op_idx, (machine, proc) in enumerate(ops):
            task_id = (j_idx, op_idx)
            tasks.append(task_id)
            job_task_list.append(task_id)
            M_t[task_id] = [machine]
            W_t[task_id] = [0]
            # p key: (task, machine, worker)
            p_dict[(task_id, machine, 0)] = float(proc)
            total_proc += proc
        job_tasks[j_idx] = job_task_list

    # precedence arcs: within each job, op_k -> op_{k+1}
    P = []
    for j_idx, tlist in job_tasks.items():
        for k in range(len(tlist) - 1):
            P.append((tlist[k], tlist[k + 1]))

    V = float(total_proc + 1)

    data = {
        "machines": machines,
        "workers": workers,
        "tasks": tasks,
        "job_tasks": job_tasks,
        "M_t": M_t,
        "W_t": W_t,
        "p": p_dict,
        "P": P,
        "V": V,
    }

    return data
