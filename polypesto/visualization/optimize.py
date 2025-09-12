from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

import pypesto.visualize as vis
from pypesto.result import Result

from polypesto.core.pypesto import has_optimization_results, get_true_param_values
from .true import plot_true_params_on_pairgrid
from .base import safe_plot

##########################
### Optimization Plots ###
##########################


@safe_plot
def plot_waterfall(result: Result, **kwargs) -> Tuple[Figure, Axes]:
    """Plots the waterfall chart.

    Args:
        result (Result): The result object containing the optimization results.

    Returns:
        (Figure, Axes): The figure and axes objects.
    """

    if not has_optimization_results(result):
        return plt.subplots()

    axes = vis.waterfall(results=result, **kwargs)
    fig = plt.gcf()
    plt.tight_layout()

    return fig, axes


@safe_plot
def plot_optimization_scatter(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, sns.PairGrid]:
    """Plots the optimization scatter.

    Args:
        result (Result): The result object containing the optimization results.
        true_params (Optional[Dict[str, float]], optional): The true parameter values. Defaults to None.

    Returns:
        (Figure, sns.PairGrid): The figure and pair grid objects.
    """

    # Return empty figure if no optimization results
    if not has_optimization_results(result):
        return plt.subplots()

    # Create the scatter plot
    kwargs.setdefault("show_bounds", True)
    grid = vis.optimization_scatter(result=result, **kwargs)
    fig = plt.gcf()

    if true_params is None:
        plt.tight_layout()
        return fig, grid

    # Get true parameter values
    true_values = get_true_param_values(result, true_params, scaled=True)
    print(true_values)
    # param_names = grid.x_vars

    # Return if no grid axes or parameter names
    if not hasattr(grid, "axes") or len(grid.axes) == 0:
        plt.tight_layout()
        return fig, grid

    plot_true_params_on_pairgrid(grid, true_values)

    plt.tight_layout()

    return fig, grid
