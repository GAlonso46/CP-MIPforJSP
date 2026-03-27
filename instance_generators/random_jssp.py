"""
Basic JSSP random instance generator.
Each job has m operations (one per machine). For each job, a random permutation of machines is chosen,
and for each operation an integer processing time uniformly sampled between 1 and 99 (inclusive) is assigned,
using Python's random module.

Main functions:
- generate_instance(j, m, seed=None): returns the instance as a dictionary.
- instance_to_string(instance): returns a readable JSON representation of the instance.

Returned instance format:
{
  "num_jobs": j,
  "num_machines": m,
  "jobs": [
    {
      "job_id": 0,
      "operations": [
        {"operation_id": 0, "machine": 2, "processing_time": 45},
        ...
      ]
    },
    ...
  ]
}

This file can also be executed as a script to generate an instance from the command line.
"""

from typing import Dict, Optional
import random
import json


def generate_instance(j: int, m: int, seed: Optional[int] = None) -> Dict:
    """Generates and returns a random JSSP instance.

    Args:
        j: number of jobs (must be > 0)
        m: number of machines (must be > 0)
        seed: optional seed for reproducibility

    Returns:
        Dictionary with the instance in the format described above.
    """
    if j <= 0 or m <= 0:
        raise ValueError("j and m must be positive integers")

    if seed is not None:
        random.seed(seed)

    instance = {"num_jobs": j, "num_machines": m, "jobs": []}
    machines = list(range(m))

    for job_id in range(j):
        perm = machines[:]  # copy
        random.shuffle(perm)
        ops = []
        for op_idx in range(m):
            proc_time = random.randint(1, 99)  # uniform 1..99
            ops.append({
                "operation_id": op_idx,
                "machine": perm[op_idx],
                "processing_time": proc_time,
            })
        instance["jobs"].append({"job_id": job_id, "operations": ops})

    return instance


def instance_to_string(instance: Dict) -> str:
    """Returns a readable JSON representation of the instance."""
    return json.dumps(instance, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Basic JSSP random generator")
    parser.add_argument("j", type=int, help="Number of jobs")
    parser.add_argument("m", type=int, help="Number of machines")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    inst = generate_instance(args.j, args.m, seed=args.seed)
    print(instance_to_string(inst))