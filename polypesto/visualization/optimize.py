from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

import pypesto.visualize as vis
from pypesto.result import Result

from polypesto.core.results import OptimizationResult
from .true import plot_true_params_on_pairgrid
from .base import safe_plot

##########################
### Optimization Plots ###
##########################


@safe_plot
def plot_waterfall(result: Result, **kwargs) -> Tuple[Figure, Axes]:

    if not hasattr(result, "optimize_result") or result.optimize_result is None:
        return plt.subplots()

    axes = vis.waterfall(results=result, **kwargs)
    fig = plt.gcf()
    plt.tight_layout()

    return fig, axes


@safe_plot
def plot_optimization_scatter(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, sns.PairGrid]:

    # Return empty figure if no optimization results
    if not hasattr(result, "optimize_result") or result.optimize_result is None:
        return plt.subplots()

    # Create the scatter plot
    kwargs.setdefault("show_bounds", True)
    grid = vis.optimization_scatter(result=result, **kwargs)
    fig = plt.gcf()

    if true_params is None:
        plt.tight_layout()
        return fig, grid

    # Get true parameter values
    opt_result = OptimizationResult(result, true_params)
    param_names = grid.x_vars
    true_values = opt_result.get_true_parameter_values(scaled=True)

    # Return if no grid axes or parameter names
    if not param_names or not hasattr(grid, "axes") or len(grid.axes) == 0:
        plt.tight_layout()
        return fig, grid

    plot_true_params_on_pairgrid(grid, true_values, param_names)

    plt.tight_layout()

    return fig, grid
