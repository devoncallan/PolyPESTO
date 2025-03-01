"""
Parameter comparison across experimental conditions.

This module provides functions for comparing parameter estimation results
across different experimental conditions and visualizing the comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any

from polypesto.core.experiments import load_all_experiments, load_experiment


def create_parameter_comparison_df(data_dir, model_name="irreversible_cpe", param_id="p_000"):
    """
    Create a comprehensive DataFrame with parameter values across all experimental conditions.
    
    Parameters
    ----------
    data_dir : str
        Path to the data directory containing experiment folders
    model_name : str, optional
        Model name to use when loading experiments
    param_id : str, optional
        Parameter set ID to use for comparison
        
    Returns
    -------
    DataFrame
        DataFrame with conditions, true parameters, estimated parameters, etc.
    """
    # Load all experiments
    experiments = load_all_experiments(data_dir, model_name)
    
    # Extract parameter values from each experiment
    param_data = []
    
    for exp_name, exp_data in experiments.items():
        try:
            # Get the conditions for this experiment - using the full path directly
            petab_dir = os.path.join(data_dir, exp_name, "petab")
            cond_path = os.path.join(petab_dir, "common", "conditions.tsv")
            
            if not os.path.isfile(cond_path):
                print(f"Conditions file not found for {exp_name}: {cond_path}")
                continue
                
            conditions = pd.read_csv(cond_path, sep='\t')
            
            # Get all condition columns except ID and name
            condition_values = {}
            for col in conditions.columns:
                if col not in ['conditionId', 'conditionName']:
                    condition_values[col] = conditions[col].values[0]
            
            # Try to get problem and result
            try:
                # Get problem information
                importer, problem = exp_data.get_problem(param_id)
                param_names = problem.x_names
                param_scales = problem.x_scales
                
                # Get result for this parameter set
                result = exp_data.get_result(param_id)
                
                # Check if the result has optimize_result
                if not hasattr(result, 'optimize_result') or result.optimize_result is None:
                    print(f"No optimization result for {exp_name}, {param_id}")
                    continue
                    
                # Get best parameters
                best_params = result.optimize_result.x
                fval = result.optimize_result.fval
                
                # Get true parameter values
                true_params = exp_data.get_true_params(param_id)
                true_param_dict = true_params.to_dict()
                
                # Create data row
                row = {
                    'experiment': exp_name,
                    'fval': fval
                }
                
                # Add condition values
                row.update(condition_values)
                
                # Add parameter values (estimated and true)
                for i, param_name in enumerate(param_names):
                    if i >= len(best_params):
                        print(f"Parameter index {i} out of range for {exp_name}, {param_id}")
                        continue
                        
                    # Estimated parameter (already in the model's scale)
                    row[f"{param_name}_est"] = best_params[i]
                    
                    # True parameter (convert to model's scale if needed)
                    if param_name in true_param_dict:
                        true_val = true_param_dict[param_name]
                        if i < len(param_scales) and param_scales[i] == "log10":
                            true_val = np.log10(true_val)
                        row[f"{param_name}_true"] = true_val
                        
                        # Calculate estimation error
                        row[f"{param_name}_error"] = best_params[i] - true_val
                        
                        # Calculate relative estimation error
                        if true_val != 0:
                            row[f"{param_name}_rel_error"] = (best_params[i] - true_val) / np.abs(true_val)
                        else:
                            row[f"{param_name}_rel_error"] = np.nan
                
                param_data.append(row)
                
            except Exception as e:
                print(f"Error processing result for {exp_name}, {param_id}: {e}")
                
        except Exception as e:
            print(f"Error processing experiment {exp_name}: {e}")
    
    # Create DataFrame
    if not param_data:
        return pd.DataFrame()
        
    return pd.DataFrame(param_data)


def compare_parameters_across_conditions(data_dir, model_name="irreversible_cpe", param_id="p_000", **kwargs):
    """
    Compare parameter estimation results across different conditions.
    
    This function loads results from multiple experiment conditions and creates
    a comprehensive comparison with visualizations.
    
    Parameters
    ----------
    data_dir : str
        Path to the data directory containing experiment folders
    model_name : str, optional
        Model name to use when loading experiments
    param_id : str, optional
        Parameter set ID to use for comparison
    **kwargs : dict
        Additional visualization options including:
        - figsize: tuple - Base figure size for plots
        - condition_col: str - Column name for condition variable
        - error_scale: str - Y-axis scale for error plots ('log' or 'linear')
        
    Returns
    -------
    tuple
        (DataFrame with comparison data, dict with visualization figures)
    """
    # Extract kwargs or use defaults
    figsize = kwargs.pop('figsize', (10, 8))
    condition_col = kwargs.pop('condition_col', 'fA0')
    error_scale = kwargs.pop('error_scale', 'log')
    
    # Load experiments and create comparison DataFrame
    comparison_df = create_parameter_comparison_df(data_dir, model_name, param_id)
    
    # If no data, return empty results
    if comparison_df.empty:
        return comparison_df, {}
    
    # Create visualizations
    figures = {}
    
    # 1. Parameter correlation heatmap
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    # Get parameter columns (estimated values)
    param_cols = [col for col in comparison_df.columns if col.endswith('_est')]
    param_names = [col.replace('_est', '') for col in param_cols]
    
    # Create correlation matrix
    if param_cols and condition_col in comparison_df.columns:
        # Include parameters and conditions in correlation
        corr_cols = [condition_col] + param_cols + ['fval']
        corr_cols = [col for col in corr_cols if col in comparison_df.columns]
        
        if corr_cols:
            corr_matrix = comparison_df[corr_cols].corr()
            
            # Plot heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
            ax1.set_title("Parameter Correlation Across Conditions")
            figures['correlation_heatmap'] = (fig1, ax1)
    
    # 2. Parameter error by condition
    fig2, ax2 = plt.subplots(figsize=figsize)
    
    # Get error columns
    error_cols = [col for col in comparison_df.columns if col.endswith('_error')]
    
    # Reshape data for plotting
    error_data = []
    for idx, row in comparison_df.iterrows():
        condition_val = row.get(condition_col, 'Unknown')
        for param, error_col in zip(param_names, error_cols):
            if error_col in row and pd.notnull(row[error_col]):
                error_data.append({
                    'Condition': condition_val,
                    'Parameter': param,
                    'Error': abs(row[error_col])
                })
    
    if error_data:
        error_df = pd.DataFrame(error_data)
        sns.boxplot(x='Condition', y='Error', hue='Parameter', data=error_df, ax=ax2)
        if error_scale == 'log':
            ax2.set_yscale('log')
        ax2.set_title('Parameter Errors by Condition')
        figures['error_boxplot'] = (fig2, ax2)
    
    # 3. Objective values by condition
    if 'fval' in comparison_df.columns and condition_col in comparison_df.columns:
        fig3, ax3 = plt.subplots(figsize=(figsize[0] * 0.8, figsize[1] * 0.8))
        if len(comparison_df) > 1:
            sns.barplot(x=condition_col, y='fval', data=comparison_df, ax=ax3)
        else:
            # For a single condition, create a simple bar chart
            ax3.bar([comparison_df[condition_col].iloc[0]], [comparison_df['fval'].iloc[0]])
            
        ax3.set_title('Objective Values by Condition')
        ax3.set_xlabel('Condition')
        ax3.set_ylabel('Objective Value')
        figures['objective_barplot'] = (fig3, ax3)
    
    # 4. True vs estimated parameters scatter plot
    if param_cols and any(col.endswith('_true') for col in comparison_df.columns):
        fig4, axes = plt.subplots(
            len(param_names), 
            1, 
            figsize=(figsize[0], figsize[1] * len(param_names) / 2),
            squeeze=False
        )
        
        for i, param in enumerate(param_names):
            true_col = f"{param}_true"
            est_col = f"{param}_est"
            if true_col in comparison_df.columns and est_col in comparison_df.columns:
                ax = axes[i, 0]
                sns.scatterplot(
                    x=true_col, 
                    y=est_col, 
                    hue=condition_col,
                    data=comparison_df, 
                    ax=ax
                )
                
                # Add identity line (perfect estimation)
                min_val = min(comparison_df[true_col].min(), comparison_df[est_col].min())
                max_val = max(comparison_df[true_col].max(), comparison_df[est_col].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                ax.set_title(f"Parameter {param}: True vs Estimated")
                ax.set_xlabel(f"True Value")
                ax.set_ylabel(f"Estimated Value")
        
        plt.tight_layout()
        figures['true_vs_estimated'] = (fig4, axes)
    
    # Apply tight layout to all figures
    for fig, _ in figures.items():
        if fig == 'true_vs_estimated':
            continue  # Already has tight_layout applied
        fig_obj = figures[fig][0]
        fig_obj.tight_layout()
    
    return comparison_df, figures