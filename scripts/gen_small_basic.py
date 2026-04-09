"""
Script to generate small basic JSSP instances for benchmarking CP and MILP models.
Generates 40 instances with specific configurations and sequential seeds starting at 2026.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Dynamically append the project root to sys.path to allow imports from sibling directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from instance_generators.random_jssp import generate_instance
from utils.convert_to_ta import write_instance_to_ta_file

def generate_small_instances() -> None:
    """
    Generates 40 small JSSP instances distributed across 4 size configurations.
    Files are saved in 'instances/basic_jssp' with names 'small01' to 'small40'.
    Seeds increment sequentially starting from 2026.
    """
    
    # Define the configurations as (num_jobs, num_machines)
    configurations: List[Tuple[int, int]] = [
        (5, 3),
        (5, 5),
        (5, 8),
        (8, 8)
    ]
    
    instances_per_config: int = 10
    current_seed: int = 2026
    instance_counter: int = 1
    output_path: str = "instances/basic_jssp"
    
    print(f"Starting generation of {len(configurations) * instances_per_config} small JSSP instances...")
    
    for jobs, machines in configurations:
        print(f"Generating 10 instances of size {jobs}x{machines}...")
        for _ in range(instances_per_config):
            # Generate the instance dictionary
            instance_data = generate_instance(j=jobs, m=machines, seed=current_seed)
            
            # Format the filename with leading zero (e.g., small01, small02)
            filename = f"small{instance_counter:02d}"
            
            # Write to file
            saved_path = write_instance_to_ta_file(
                instance=instance_data, 
                path=output_path, 
                filename=filename
            )
            
            # Increment seed and counter for the next iteration
            current_seed += 1
            instance_counter += 1
            
    print(f"Successfully generated all instances in: {output_path}")

if __name__ == "__main__":
    generate_small_instances()