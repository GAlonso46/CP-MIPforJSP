"""
Utilities to convert JSSP instances (as produced by generate_instance) into the "ta" text format
used by the provided example files (e.g. `ta11`).

The format produced by `convert_instance_to_ta_format` is:

<num_jobs> <num_machines>
<m0_0> <p0_0> <m0_1> <p0_1> ... <m0_{m-1}> <p0_{m-1}>
<m1_0> <p1_0> <m1_1> <p1_1> ... <m1_{m-1}> <p1_{m-1}>
...

Where each line after the header corresponds to one job and contains `num_machines` pairs
of integers: machine_id processing_time.

This module exposes two functions:
- convert_instance_to_ta_format(instance) -> str
- write_instance_to_ta_file(instance, path, filename) -> str  (returns the written file path)
"""

from typing import Dict
from pathlib import Path
import os


def convert_instance_to_ta_format(instance: Dict) -> str:
    """Convert a JSSP instance dictionary to the TA text format and return it as a string.

    Args:
        instance: dictionary with keys "num_jobs", "num_machines" and "jobs" as produced
                  by `generate_instance` in `random_jssp.py`.

    Returns:
        A string containing the instance in TA format.

    Raises:
        ValueError: if the instance structure is invalid or job operations do not match
                    the declared number of machines.
    """
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dictionary")

    try:
        num_jobs = int(instance["num_jobs"])
        num_machines = int(instance["num_machines"])
        jobs = instance["jobs"]
    except Exception as exc:
        raise ValueError("instance missing required keys: num_jobs, num_machines, jobs") from exc

    lines = []
    # header
    lines.append(f"{num_jobs} {num_machines}")

    for job in jobs:
        ops = job.get("operations")
        if ops is None or len(ops) != num_machines:
            raise ValueError(f"Each job must have exactly {num_machines} operations")

        # For each operation preserve the order of operations (operation_id ascending)
        # and emit pairs: machine processing_time
        # Ensure operations are sorted by operation_id in case they are out of order
        sorted_ops = sorted(ops, key=lambda o: int(o.get("operation_id", 0)))
        pair_strs = []
        for op in sorted_ops:
            machine = int(op["machine"])
            proc = int(op["processing_time"])
            pair_strs.append(f"{machine} {proc}")

        # Join pairs with a single space between them (pairs themselves contain a space).
        line = " ".join(pair_strs)
        lines.append(line)

    # Join all lines with newline and ensure trailing newline at the end of file
    content = "\n".join(lines) + "\n"
    return content


def write_instance_to_ta_file(instance: Dict, path: str, filename: str) -> str:
    """Write the given instance to a file in TA format.

    Args:
        instance: JSSP instance dictionary (same format as input of convert_instance_to_ta_format).
        path: directory path where the file will be created. If it does not exist it will be created.
        filename: name of the file to create (e.g. 'ta11').

    Returns:
        The full path to the written file as a string.
    """
    content = convert_instance_to_ta_format(instance)

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    # Write using default system encoding; overwrite if exists
    with out_path.open("w", encoding="utf-8") as f:
        f.write(content)

    return str(out_path)
