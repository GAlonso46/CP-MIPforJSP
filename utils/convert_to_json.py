"""
Module for exporting Extended Job Shop Scheduling Problem (JSSP) instances 
to JSON format for persistence and interoperability.
"""

import json
import os

def save_extended_jssp_instance(instance, folder_path, file_name):
    """
    Saves or overwrites an extended JSSP instance dictionary into a JSON file.

    Args:
        instance (dict): Dictionary containing the following keys:
            - machines: list of machine ids (ints)
            - workers: list of worker ids (ints)
            - tasks: list of task ids (tuples: (job_id, op_idx))
            - job_tasks: mapping job_id -> list of task ids
            - M_t: mapping task -> list of eligible machines
            - W_m: mapping worker -> list of machines the worker can operate
            - p: mapping (task, machine, worker) -> processing_time (int)
            - P: list of precedence arcs (task_prev, task_next)
        folder_path (str): Path to the directory where the file will be saved.
        file_name (str): Name of the output file (e.g., 'instance_01.json').

    Returns:
        str: The full path to the saved file.
    """
    
    # Ensure the directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    full_path = os.path.join(folder_path, file_name)

    # JSON does not support tuples as keys or elements. 
    # We must convert tuple-based keys and lists to string representations.
    serializable_instance = {
        "machines": instance["machines"],
        "workers": instance.get("workers", [0]),
        # Convert task tuples (job_id, op_idx) to strings "job_id_op_idx"
        "tasks": [f"{t[0]}_{t[1]}" for t in instance["tasks"]],
        "job_tasks": {str(k): [f"{t[0]}_{t[1]}" for t in v] 
                      for k, v in instance["job_tasks"].items()},
        "M_t": {f"{t[0]}_{t[1]}": v for t, v in instance["M_t"].items()},
        "W_m": {str(k): v for k, v in instance.get("W_m", {}).items()},
        # Convert (task, machine, worker) keys to a single string key
        "p": {f"{t[0]}_{t[1]}|{m}|{w}": val 
              for (t, m, w), val in instance["p"].items()},
        # Convert precedence arcs (task_prev, task_next) to string pairs
        "P": [[f"{t1[0]}_{t1[1]}", f"{t2[0]}_{t2[1]}"] for t1, t2 in instance["P"]]
    }

    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_instance, f, indent=4)
    
    print(f"Instance successfully saved to: {full_path}")
    return full_path