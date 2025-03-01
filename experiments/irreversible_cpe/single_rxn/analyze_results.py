"""
Example script for analyzing parameter estimation results using the new
refactored visualization and analysis API.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the PolyPESTO root directory to the Python path
sys.path.insert(0, os.path.abspath('/PolyPESTO'))

# Import PolyPESTO visualization and analysis modules
from polypesto.core.experiments import load_experiment
from polypesto.visualization import (
    plot_waterfall,
    plot_optimization_scatter,
    plot_profiles,
    plot_parameter_traces,
    plot_confidence_intervals,
    visualize_parameter_estimation
)
from polypesto.analysis import (
    compare_parameters_across_conditions,
    create_parameter_comparison_df
)

# Path to experiment data
DATA_DIR = Path(__file__).parent / "data"
MODEL_NAME = "irreversible_cpe"
PARAM_ID = "p_000"
CONDITION = "fA0_0.50"

def analyze_single_experiment():
    """Analyze a single experiment using the visualization functions."""
    # Load the experiment 
    experiment = load_experiment(DATA_DIR / CONDITION, MODEL_NAME)
    
    # Get the PyPESTO result for parameter set p_000
    result = experiment.get_result(PARAM_ID) 
    
    # Get true parameters for comparison
    true_params = experiment.get_true_params(PARAM_ID)
    
    # Generate all visualizations with a single function call
    vis_results = visualize_parameter_estimation(
        result=result,
        true_params=true_params,
        plots=['waterfall', 'scatter', 'profiles', 'traces', 'intervals'],
        # Additional customization via kwargs
        waterfall_kwargs={'scale_y': 'log10'},
        scatter_kwargs={'diag_kind': 'kde'},
        profiles_kwargs={'show_bounds': True},
        intervals_kwargs={"alpha": [90, 95, 99]}
    )
    
    # Display and save each visualization
    for plot_type, (fig, ax) in vis_results.items():
        # Set a descriptive title
        if hasattr(fig, 'suptitle'):
            fig.suptitle(f"{plot_type.capitalize()} - {CONDITION}")
        
        # Save the figure
        output_path = f"parameter_{plot_type}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    
    return vis_results

def compare_across_conditions():
    """Compare parameter estimation results across different conditions."""
    # Create comparison dataframe
    comparison_df = create_parameter_comparison_df(
        data_dir=DATA_DIR,
        model_name=MODEL_NAME,
        param_id=PARAM_ID
    )
    
    # Generate comparison visualizations
    df, figures = compare_parameters_across_conditions(
        data_dir=DATA_DIR,
        model_name=MODEL_NAME,
        param_id=PARAM_ID,
        figsize=(10, 8)
    )
    
    # Save each comparison figure
    for fig_name, (fig, _) in figures.items():
        output_path = f"parameter_{fig_name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    
    return comparison_df, figures

if __name__ == "__main__":  
    # Analyze a single condition
    print("Analyzing single experiment...")
    vis_results = analyze_single_experiment()
    
    # Compare across conditions
    # print("\nComparing across conditions...")
    # comparison_df, comparison_figures = compare_across_conditions()
    
    print("\nAnalysis complete!")