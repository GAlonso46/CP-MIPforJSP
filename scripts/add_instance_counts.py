"""
Add num_machines, num_jobs and num_workers fields to every JSON instance under instances/.

This script walks the `instances/` folder recursively, opens each .json file, computes
- num_machines => len(data.get('machines', []))
- num_workers  => len(data.get('workers', []))
- num_jobs     => number of entries in data.get('job_tasks', {})

The script updates (or inserts) these three keys at the top-level of each JSON file and
writes the file back. Nothing else in the data structure is modified.

Usage: python scripts/add_instance_counts.py [--instances DIR]
"""
from pathlib import Path
import json
import argparse
import tempfile
import shutil


def process_file(path: Path, dry_run: bool = False) -> bool:
    """Load JSON, compute counts, update file in place. Returns True on success."""
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping {path}: failed to read JSON ({e})")
        return False

    # Compute counts from existing fields
    machines = data.get('machines', [])
    workers = data.get('workers', [])
    job_tasks = data.get('job_tasks', {})

    try:
        num_machines = int(len(machines))
    except Exception:
        num_machines = 0
    try:
        num_workers = int(len(workers))
    except Exception:
        num_workers = 0
    try:
        # job_tasks is expected to be a mapping job_id -> list
        num_jobs = int(len(job_tasks))
    except Exception:
        num_jobs = 0

    changed = False
    # Only set the keys if missing or different (avoid unnecessary file writes)
    if data.get('num_machines') != num_machines:
        data['num_machines'] = num_machines
        changed = True
    if data.get('num_workers') != num_workers:
        data['num_workers'] = num_workers
        changed = True
    if data.get('num_jobs') != num_jobs:
        data['num_jobs'] = num_jobs
        changed = True

    if not changed:
        print(f"No change for {path}")
        return True

    if dry_run:
        print(f"Would update {path}: num_machines={num_machines}, num_workers={num_workers}, num_jobs={num_jobs}")
        return True

    # Write safely to a temp file then replace original
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tf:
            json.dump(data, tf, indent=2, ensure_ascii=False)
            temp_name = tf.name
        shutil.move(temp_name, str(path))
        print(f"Updated {path}: num_machines={num_machines}, num_workers={num_workers}, num_jobs={num_jobs}")
        return True
    except Exception as e:
        print(f"Failed to write {path}: {e}")
        return False


def run(instances_dir: Path, dry_run: bool = False):
    instances_dir = Path(instances_dir)
    if not instances_dir.exists():
        raise FileNotFoundError(f"Instances directory not found: {instances_dir}")

    files = list(instances_dir.rglob('*.json'))
    print(f"Found {len(files)} json files under {instances_dir}")

    success = 0
    failed = 0
    skipped = 0

    for p in files:
        ok = process_file(p, dry_run=dry_run)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"Done. success={success} failed={failed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add num_machines/num_jobs/num_workers to instance JSONs')
    parser.add_argument('--instances', type=Path, default=Path(__file__).parent.parent / 'instances', help='Path to instances root')
    parser.add_argument('--dry-run', action='store_true', help='Do not write changes, only report')
    args = parser.parse_args()

    run(args.instances, dry_run=args.dry_run)
