import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.gantt_diagrams import generate_comparison_from_json

def test_grantt():
    results_root = Path(__file__).parent / 'test_results'
    folder_results_root = results_root / 'basic_jssp'
    instance_results_root = folder_results_root / 'small01.json'

    generate_comparison_from_json(
        instance_results_root, 
        folder_results_root, 
        "comparison_test.pdf", 
        'scip')
    
if __name__ == '__main__':
    test_grantt()    