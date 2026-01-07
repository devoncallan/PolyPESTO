from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pypesto.visualize as vis  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from polypesto.core.pypesto import Result, get_true_param_values, has_sampling_results

from .base import safe_plot
from .true import (
    draw_true_param_marker,
    plot_true_params_on_pairgrid,
    plot_true_params_on_trace,
)

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
        fig, axs = plt.subplots()
        return fig, [axs]

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

    def log_to_actual(x, pos):
        return f"{10**x:.3f}"

    # Return if no true values or no valid grid
    # if not true_params or not hasattr(grid, "axes") or len(grid.axes) == 0:
    #     # print("Hiya")
    #     # print("Formatting axes...")
    #     # grid.axes[1,0].set_xlim(np.log10([0.35, 0.45]))
    #     # grid.axes[1,0].set_ylim(np.log10([0.5, 0.7]))
    #     # grid.axes[1,0].xaxis.set_major_formatter(FuncFormatter(log_to_actual))
    #     # grid.axes[1,0].yaxis.set_major_formatter(FuncFormatter(log_to_actual))
    #     grid.axes[1, 0].set_xticks(np.log10([0.3, 0.35, 0.4, 0.45, 0.5]))
    #     grid.axes[1, 0].set_yticks(np.log10([0.5, 0.55, 0.6, 0.65, 0.7]))
    #     grid.axes[0, 1].set_xticks(np.log10([0.5, 0.55, 0.6, 0.65, 0.7]))
    #     grid.axes[0, 1].set_yticks(np.log10([0.3, 0.35, 0.4, 0.45, 0.5]))
    #     # print("Formatting axes...")

    #     for ax in grid.axes.flatten():
    #         ax: Axes = ax
    #         # ax.xaxis.set_ticks(np.log10([0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]))
    #         # ax.xaxis.set_ticks(np.log10([0.3, 0.35, 0.4, 0.45, 0.5]))
    #         ax.xaxis.set_major_formatter(FuncFormatter(log_to_actual))
    #         ax.yaxis.set_major_formatter(FuncFormatter(log_to_actual))
        #     print("Axes:", ax)

        # ax.set_xlim(np.log10([0.3, 0.7]))
        # ax.set_ylim(np.log10([0.3, 0.7]))
        # ax.set_xlim(0.30, 0.70)
        # ax.set_ylim(0.30, 0.70)
        # ax.set_xticks([0.35, ])
        # plt.tight_layout()
        # return fig, grid

    true_values = get_true_param_values(result, true_params, scaled=True)

    plot_true_params_on_pairgrid(grid, true_values)

    # Flatten the 2D array of axes
    # print("Formatting axes...")
    # for ax in grid.axes.flatten():
    #     ax: Axes = ax
    #     print("Axes:", ax)
    #     ax.xaxis.set_major_formatter(FuncFormatter(log_to_actual))
    #     ax.yaxis.set_major_formatter(FuncFormatter(log_to_actual))

    plt.tight_layout()
    return fig, grid
