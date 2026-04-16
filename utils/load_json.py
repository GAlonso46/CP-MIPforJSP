import json
import os

def load_extended_jssp_instance(file_path):
    """
    Loads a JSON file and reconstructs the original Python types (tuples and ints)
    required by the MILP and CP optimization models.

    Args:
        file_path (str): Path to the .json file.
    
    Returns:
        dict: The reconstructed instance object.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def to_t(s): 
        parts = s.split('_')
        return (int(parts[0]), int(parts[1]))
    
    workers = data.get("workers") or []

    res = {
        "machines": data["machines"],
        "workers": workers,
        "tasks": [to_t(t) for t in data["tasks"]],
        "job_tasks": {int(j): [to_t(t) for t in ts] for j, ts in data["job_tasks"].items()},
        "P": [(to_t(p[0]), to_t(p[1])) for p in data["P"]],
        # Temporal parameters
        "L": {},
        "release_dates": {int(j): v for j, v in data["release_dates"].items()},
        "deadlines": {int(j): v for j, v in data["deadlines"].items()},
        "M_t": {to_t(t): ms for t, ms in data["M_t"].items()},
        "W_m": {int(w): ms for w, ms in data["W_m"].items()},
        "p": {},
        "s": {}
    }

    # Reconstruct L: "job_op|job_op" -> ((job, op), (job, op))
    for k, v in data.get("L", {}).items():
        t1_s, t2_s = k.split('|')
        res["L"][(to_t(t1_s), to_t(t2_s))] = v

    # Reconstruct p: "job_op|m|w" -> ((job, op), m, w)
    for k, v in data.get("p", {}).items():
        t_s, m_s, w_s = k.split('|')
        actual_w = None if w_s == 'null' else int(w_s)
        res["p"][(to_t(t_s), int(m_s), actual_w)] = v

    # Reconstruct s: "job_i_op_i|job_k_op_k|m" -> (task_i, task_k, m)
    for k, v in data.get("s", {}).items():
        ti_s, tk_s, m_s = k.split('|')
        res["s"][(to_t(ti_s), to_t(tk_s), int(m_s))] = v

    return res