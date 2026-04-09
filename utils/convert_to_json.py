"""
Module for exporting Extended Job Shop Scheduling Problem (JSSP) instances 
to JSON format for persistence and interoperability.
"""

import json
import os

def save_extended_jssp_instance(instance, folder_path, file_name):
    """
    Saves an extended JSSP instance to a JSON file, converting complex keys 
    (tuples) into strings for JSON compatibility.

    Args:
        instance (dict): Instance data containing 'machines', 'workers', 'tasks',
            'job_tasks', 'P', 'L', 'release_dates', 'deadlines', 'M_t', 'W_m', 'p', and 's'.
        folder_path (str): Directory to store the file.
        file_name (str): Name of the file.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    full_path = os.path.join(folder_path, file_name)

    # Helper to convert task tuple (job, op) to "job_op" string
    def t_str(t): return f"{t[0]}_{t[1]}"

    serializable = {
        "machines": instance["machines"],
        "workers": instance.get("workers", [0]),
        "tasks": [t_str(t) for t in instance["tasks"]],
        "job_tasks": {str(j): [t_str(t) for t in ts] for j, ts in instance["job_tasks"].items()},
        "P": [[t_str(t1), t_str(t2)] for t1, t2 in instance.get("P", [])],
        # Temporal parameters (L, release_dates, deadlines)
        "L": {f"{t_str(pair[0])}|{t_str(pair[1])}": int(v) for pair, v in instance.get("L", {}).items()},
        "release_dates": {str(j): int(v) for j, v in instance.get("release_dates", {}).items()},
        "deadlines": {str(j): int(v) for j, v in instance.get("deadlines", {}).items()},
        # Mappings
        "M_t": {t_str(t): ms for t, ms in instance["M_t"].items()},
        "W_m": {str(w): ms for w, ms in instance.get("W_m", {}).items()},
        # Processing times: ((job, op), m, w) -> "job_op|m|w"
        "p": {f"{t_str(k[0])}|{k[1]}|{k[2]}": int(v) for k, v in instance["p"].items()},
        # Setup times: ((job_i, op_i), (job_k, op_k), m) -> "job_i_op_i|job_k_op_k|m"
        "s": {f"{t_str(k[0])}|{t_str(k[1])}|{k[2]}": int(v) for k, v in instance.get("s", {}).items()}
    }
    # Include metadata if present
    if "metadata" in instance:
        serializable["metadata"] = instance["metadata"]

    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=4)
    
    return full_path