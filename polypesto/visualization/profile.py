from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import pypesto.visualize as vis
from pypesto.result import Result

# Import our result handlers
from polypesto.core.pypesto import has_profile_results, get_true_param_values
from .true import plot_true_params_on_distribution
from .base import safe_plot

#######################
### Profiling Plots ###
#######################


@safe_plot
def plot_profiles(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """Plots the profiles of the parameters.

    Args:
        result (Result): The result object containing the optimization results.
        true_params (Optional[Dict[str, float]], optional): The true parameter values. Defaults to None.

    Returns:
        (Figure, Axes): The figure and axes objects.
    """

    if not has_profile_results(result):
        return plt.subplots()

    kwargs.setdefault("show_bounds", True)
    axs = vis.profiles(result, **kwargs)
    fig = plt.gcf()

    if true_params is None:
        plt.tight_layout()
        return fig, axs

    true_values = get_true_param_values(result, true_params, scaled=True)

    plot_true_params_on_distribution(axs, true_values)
    plt.tight_layout()

    return fig, axs
