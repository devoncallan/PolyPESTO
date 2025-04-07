from typing import Optional, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

import pypesto.visualize as vis
from pypesto.result import Result

# Import our result handlers
from polypesto.core.results import SamplingResult
from .true import (
    plot_true_params_on_pairgrid,
    plot_true_params_on_trace,
    draw_true_param_marker,
)

######################
### Sampling Plots ###
######################


def plot_parameter_traces(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, List[Axes]]:

    # Create sampling result handler
    sampling_result = SamplingResult(result, true_params)

    # Return empty figure if no sampling results
    if not sampling_result.has_sampling_results():
        return plt.subplots()

    axes = vis.sampling_parameter_traces(result=result, **kwargs)
    fig = plt.gcf()

    # Skip if no true values
    true_values = sampling_result.get_true_parameter_values(scaled=True)
    if not true_values:
        plt.tight_layout()
        return fig, axes

    # Convert axes to list for iteration
    axes_list = (
        [axes]
        if not isinstance(axes, (list, np.ndarray))
        else axes.flatten() if isinstance(axes, np.ndarray) else axes
    )

    param_names = sampling_result.param_names
    plot_true_params_on_trace(axes_list, true_values, param_names)

    plt.tight_layout()
    return fig, axes


def plot_confidence_intervals(result, true_params=None, **kwargs):

    # Create sampling result handler
    sampling_result = SamplingResult(result, true_params)

    # Return empty figure if no results
    if not sampling_result.has_sampling_results():
        return plt.subplots()

    kwargs.setdefault("alpha", [90, 95, 99])
    kwargs.setdefault("show_bounds", True)

    # Plot confidence intervals
    ax = vis.sampling_parameter_cis(result=result, **kwargs)
    fig = plt.gcf()

    # Get true parameter values
    true_values = sampling_result.get_true_parameter_values(scaled=True)
    if not true_values:
        plt.tight_layout()
        return fig, ax

    # Get parameter positions from plot
    y_ticks = ax.get_yticks()
    y_labels = [label.get_text() for label in ax.get_yticklabels()]

    # Collect parameters with true values
    true_x_vals = []
    true_y_pos = []
    for i, param_id in enumerate(y_labels):
        if param_id in true_values:
            true_x_vals.append(true_values[param_id])
            true_y_pos.append(y_ticks[i])

    # Skip if no matching parameters
    if not true_x_vals:
        plt.tight_layout()
        return fig, ax

    # Plot true values
    draw_true_param_marker(ax, true_x_vals, true_y_pos)

    plt.tight_layout()
    return fig, ax


def plot_sampling_scatter(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, sns.PairGrid]:

    # Create sampling result handler
    sampling_result = SamplingResult(result, true_params)

    # Return empty figure if no sampling results
    if not sampling_result.has_sampling_results():
        return plt.subplots()

    # Create scatter plot
    grid = vis.sampling_scatter(result=result, **kwargs)
    fig = plt.gcf()

    true_values = sampling_result.get_true_parameter_values(scaled=True)
    param_names = grid.x_vars

    # Return if no true values or no valid grid
    if not true_values or not hasattr(grid, "axes") or len(grid.axes) == 0:
        plt.tight_layout()
        return fig, grid

    plot_true_params_on_pairgrid(grid, true_values, param_names)

    plt.tight_layout()
    return fig, grid
