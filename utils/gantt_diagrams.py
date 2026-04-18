import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Any

def _format_for_plotting(solution_data: dict) -> List[Dict[str, Any]]:
    """
    Transforms the solver-agnostic solution dictionary into a flat list 
    suitable for Gantt chart generation.
    """
    if not solution_data or "tasks" not in solution_data:
        return []
        
    formatted = []
    tasks = solution_data.get("tasks", {})
    modes = solution_data.get("selected_modes", [])
    
    # Create a mapping of task -> (machine, worker) for O(1) lookups
    mode_map = {t: (m, w) for t, m, w in modes}
    
    for t, times in tasks.items():
        m, w = mode_map.get(t, ("Unknown", None))
        start = times["start"]
        duration = times["end"] - start
        
        # Extract Job ID from Task ID to assign consistent colors.
        # This assumes task strings are formatted like 'Job1_Task2' or 'J1_01'.
        # Adjust the split logic if your task IDs follow a different convention.
        job_id = str(t).split('_')[0] if '_' in str(t) else "Unknown_Job"
        
        formatted.append({
            "job": job_id,
            "task": str(t),
            "machine": str(m),
            "start": start,
            "duration": duration
        })
        
    return formatted

def _get_color_map(solutions: List[List[Dict[str, Any]]]):
    """Generates a unique color map per job for consistency across plots."""
    unique_jobs = set()
    for sol in solutions:
        for op in sol:
            unique_jobs.add(op['job'])
            
    cmap = plt.get_cmap('tab20')
    # Sort jobs so colors are assigned deterministically
    return {job: cmap(i % 20) for i, job in enumerate(sorted(unique_jobs))}

def plot_gantt_single(solution_dict: dict, title: str, output_path: str):
    """
    Generates a Gantt chart for a single solution object and saves it.
    """
    solution = _format_for_plotting(solution_dict)
    if not solution:
        print(f"Warning: No valid solution data to plot for {title}.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = _get_color_map([solution])
    
    for op in solution:
        ax.barh(
            y=op['machine'], 
            width=op['duration'], 
            left=op['start'], 
            color=colors[op['job']], 
            edgecolor='black',
            height=0.6,
            alpha=0.9
        )
        ax.text(
            x=op['start'] + op['duration'] / 2, 
            y=op['machine'], 
            s=op['task'], 
            va='center', 
            ha='center', 
            color='white', 
            fontsize=8, 
            fontweight='bold'
        )

    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    legend_patches = [mpatches.Patch(color=color, label=f'{job}') for job, color in colors.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, format=output_path.split('.')[-1], dpi=300)
    plt.close()

def plot_gantt_comparison(sol_dict_1: dict, sol_dict_2: dict, title: str, output_path: str, label1="CP", label2="MILP"):
    """
    Generates a comparative Gantt chart by splitting each machine's row.
    Model 1 is displayed on the top half, Model 2 on the bottom half.
    """
    sol_1 = _format_for_plotting(sol_dict_1)
    sol_2 = _format_for_plotting(sol_dict_2)
    
    if not sol_1 and not sol_2:
        print(f"Warning: Neither model has valid solution data for {title}.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = _get_color_map([sol_1, sol_2])
    
    # Gather all unique machines from both solutions
    machines = sorted(list(set(op['machine'] for op in sol_1 + sol_2)))
    machine_to_y = {m: i for i, m in enumerate(machines)}
    
    bar_height = 0.35
    offset = 0.2

    def draw_solution(solution, current_offset, hatch_pattern, label_prefix):
        for op in solution:
            y_base = machine_to_y[op['machine']]
            ax.barh(
                y=y_base + current_offset, 
                width=op['duration'], 
                left=op['start'], 
                color=colors[op['job']], 
                edgecolor='black',
                height=bar_height,
                hatch=hatch_pattern,
                alpha=0.8
            )
            ax.text(
                x=op['start'] + op['duration'] / 2, 
                y=y_base + current_offset, 
                s=op['task'], 
                va='center', 
                ha='center', 
                color='black' if hatch_pattern else 'white', 
                fontsize=7, 
                fontweight='bold'
            )

    # Draw Model 1 (Top, no pattern) and Model 2 (Bottom, hatched pattern)
    draw_solution(sol_1, offset, None, label1)
    draw_solution(sol_2, -offset, '///', label2)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add horizontal dividing lines between machines
    for i in range(len(machines) - 1):
        ax.axhline(y=i + 0.5, color='gray', linestyle='-', alpha=0.3)

    # Construct dual legend
    legend_patches = [mpatches.Patch(color=color, label=f'{job}') for job, color in colors.items()]
    legend_patches.append(mpatches.Patch(facecolor='white', edgecolor='black', label=f'Paradigm: {label1}'))
    legend_patches.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label=f'Paradigm: {label2}'))
    
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, format=output_path.split('.')[-1], dpi=300)
    plt.close()

def generate_comparison_from_json(filepath: str, output_dir: str, filename: str, mip_key: str = 'scip'):
    """
    Reads a saved test result JSON, extracts the CP and specified MILP solutions, 
    and generates a comparative Gantt chart.
    
    Args:
        filepath: Path to the JSON file containing solver results.
        output_dir: Directory where the generated diagram should be saved.
        filename: Name of the output image file (e.g., 'comparison.pdf').
        mip_key: The key in the JSON corresponding to the MILP solver (e.g., 'scip', 'highs', 'milp').
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {filepath}: {e}")
        return

    # Extract instance name for the title
    instance_name = Path(data.get('instance', 'Unknown_Instance')).stem
    
    # Safely extract solutions
    cp_data = data.get('cp') or {}
    mip_data = data.get(mip_key) or {}
    
    cp_solution = cp_data.get('solution', {})
    mip_solution = mip_data.get('solution', {})
    
    if not cp_solution and not mip_solution:
        print(f"Skipping {instance_name}: Neither CP nor {mip_key.upper()} found feasible solutions.")
        return

    title = f"Gantt Comparison: CP vs {mip_key.upper()} - {instance_name}"
    
    # Generate the plot
    plot_gantt_comparison(
        sol_dict_1=cp_solution, 
        sol_dict_2=mip_solution, 
        title=title, 
        output_path=out_path,
        label1="CP",
        label2=mip_key.upper()
    )
    print(f"Successfully generated diagram: {out_path}")