"""
Script to generate extended JSSP variant instances from basic JSSP base files.
Reads basic instances in a specific order (small, la, ta), applies variant generators,
and saves the results in their respective directories with sequential seeds starting at 2066.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Callable

# Dynamically append the project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.load_ta import parse_ta_file
from utils.convert_to_json import save_extended_jssp_instance
from instance_generators.random_jssp_variants import (
    generate_fjssp_variant,
    generate_timelags_variant,
    generate_release_dates_variant,
    generate_sdst_uniform_variant,
    generate_deadlines_variant,
    generate_dual_resources_variant
)

def build_target_file_list(base_dir: Path) -> List[Path]:
    """
    Constructs the exact list of file paths in the following order:
    1. small01 to small40
    2. la01 to la40
    3. ta11 to ta50
    """
    files: List[Path] = []
    
    # 1. smallxx (small01 -> small40)
    for i in range(1, 41):
        files.append(base_dir / f"small{i:02d}")
        
    # 2. laxx (la01 -> la40)
    for i in range(1, 41):
        files.append(base_dir / f"la{i:02d}")
        
    # 3. taxx (ta11 -> ta50)
    for i in range(11, 51):
        files.append(base_dir / f"ta{i}")
        
    return files

def generate_all_variants() -> None:
    """
    Main pipeline to parse base instances, generate variants, and save them.
    Iterates over variants, then over instances, incrementing the seed continuously.
    """
    instances_dir = PROJECT_ROOT / "instances"
    basic_dir = instances_dir / "basic_jssp"
    
    # Configuration matrix for each variant
    # Contains the directory/prefix, the generator function, and default hyperparameters
    variants_config = [
        {
            "prefix": "f_jssp", 
            "func": generate_fjssp_variant, 
            "kwargs": {"min_m": 1, "max_m": 3}
        },
        {
            "prefix": "t_jssp", 
            "func": generate_timelags_variant, 
            "kwargs": {"density": 0.3, "min_lag": 1, "max_lag": 10}
        },
        {
            "prefix": "r_jssp", 
            "func": generate_release_dates_variant, 
            "kwargs": {"job_prob": 0.4, "max_r_ratio": 0.5}
        },
        {
            "prefix": "d_jssp", 
            "func": generate_deadlines_variant, 
            "kwargs": {"deadline_density": 0.3, "gamma_min": 1.3, "gamma_max": 1.6, "noise_ratio": 0.2}
        },
        {
            "prefix": "sdst_fjjsp", 
            "func": generate_sdst_uniform_variant, 
            "kwargs": {"alpha": 0.5}
        },
        {
            "prefix": "drc_jssp", 
            "func": generate_dual_resources_variant, 
            "kwargs": {"rho_workers": 1.0, "compatibility_prob": 0.3, "delta": 0.2}
        }
    ]

    # 1. Gather files in strictly specified order
    expected_files = build_target_file_list(basic_dir)
    valid_files = [f for f in expected_files if f.exists()]
    
    if not valid_files:
        print(f"Error: No basic instances found in {basic_dir}")
        return
        
    print(f"Found {len(valid_files)} basic instances. Parsing data...")
    
    # Parse files once and cache in memory to avoid redundant disk reads
    parsed_instances = []
    for file_path in valid_files:
        try:
            parsed_data = parse_ta_file(str(file_path))
            parsed_instances.append(parsed_data)
        except Exception as e:
            print(f"Failed to parse {file_path.name}: {e}")
            return

    # 2. Generation Loop Initialization
    current_seed: int = 2066
    
    # Outer loop: Iterate over each variant
    for variant in variants_config:
        prefix = variant["prefix"]
        gen_func = variant["func"]
        kwargs = variant["kwargs"]
        
        output_folder = str(instances_dir / prefix)
        print(f"\n--- Generating {prefix} variant ---")
        
        # Inner loop: Iterate over the ordered base instances
        for idx, base_instance in enumerate(parsed_instances, start=1):
            
            # Generate the new extended instance
            extended_instance = gen_func(data=base_instance, seed=current_seed, **kwargs)
            
            # Format filename with 3-digit zero-padding (e.g., d_jssp001.json)
            file_name = f"{prefix}{idx:03d}.json"
            
            # Save to disk
            saved_path = save_extended_jssp_instance(
                instance=extended_instance, 
                folder_path=output_folder, 
                file_name=file_name
            )
            
            # Increment the seed globally
            current_seed += 1
            
        print(f"Successfully generated {len(parsed_instances)} files for {prefix} in {output_folder}")

if __name__ == "__main__":
    generate_all_variants()