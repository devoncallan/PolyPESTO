from typing import Optional, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

import pypesto.visualize as vis
from pypesto.result import Result

from polypesto.core.pypesto import has_sampling_results, get_true_param_values
from .true import (
    plot_true_params_on_pairgrid,
    plot_true_params_on_trace,
    draw_true_param_marker,
)
from .base import safe_plot

######################
### Sampling Plots ###
######################


@safe_plot
def plot_parameter_traces(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, List[Axes]]:
    """Plots the parameter traces.

    Args:
        result (Result): The result object containing the sampling results.
        true_params (Optional[Dict[str, float]], optional): The true parameter values. Defaults to None.

    Returns:
        (Figure, List[Axes]): The figure and axes objects.
    """

    # Return empty figure if no sampling results
    if not has_sampling_results(result):
        return plt.subplots()

    axes = vis.sampling_parameter_traces(result=result, **kwargs)
    fig = plt.gcf()

    # Skip if no true values
    true_values = get_true_param_values(result, true_params, scaled=True)
    if not true_values:
        plt.tight_layout()
        return fig, axes

    # Convert axes to list for iteration
    axes_list = (
        [axes]
        if not isinstance(axes, (list, np.ndarray))
        else axes.flatten() if isinstance(axes, np.ndarray) else axes
    )

    plot_true_params_on_trace(axes_list, true_values)

    plt.tight_layout()
    return fig, axes


@safe_plot
def plot_confidence_intervals(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """Plots the confidence intervals.

    Args:
        result (Result): The result object containing the sampling results.
        true_params (Optional[Dict[str, float]], optional): The true parameter values. Defaults to None.

    Returns:
        (Figure, Axes): The figure and axes objects.
    """

    # Return empty figure if no results
    if not has_sampling_results(result):
        return plt.subplots()

    kwargs.setdefault("alpha", [90, 95, 99])
    # Plot confidence intervals
    ax = vis.sampling_parameter_cis(result=result, **kwargs)
    ax.set_xlim(min(result.problem.lb), max(result.problem.ub))
    fig = plt.gcf()

    # Get true parameter values
    if not true_params:
        plt.tight_layout()
        return fig, ax

    true_values = get_true_param_values(result, true_params, scaled=True)

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


@safe_plot
def plot_sampling_scatter(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, sns.PairGrid]:
    """Plots the sampling scatter.

    Args:
        result (Result): The result object containing the sampling results.
        true_params (Optional[Dict[str, float]], optional): The true parameter values. Defaults to None.

    Returns:
        (Figure, sns.PairGrid): The figure and pair grid objects.
    """

    # Return empty figure if no sampling results
    if not has_sampling_results(result):
        return plt.subplots()

    # Create scatter plot
    grid = vis.sampling_scatter(result=result, **kwargs)
    fig = plt.gcf()

    # Return if no true values or no valid grid
    if not true_params or not hasattr(grid, "axes") or len(grid.axes) == 0:
        plt.tight_layout()
        return fig, grid

    true_values = get_true_param_values(result, true_params, scaled=True)

    plot_true_params_on_pairgrid(grid, true_values)

    plt.tight_layout()
    return fig, grid
