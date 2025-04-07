from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import pypesto.visualize as vis
from pypesto.result import Result

# Import our result handlers
from polypesto.core.results import ProfileResult
from .true import plot_true_params_on_distribution
from .base import safe_plot

#######################
### Profiling Plots ###
#######################


@safe_plot
def plot_profiles(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
) -> Tuple[Figure, Axes]:

    kwargs.setdefault("show_bounds", True)
    axs = vis.profiles(result, **kwargs)
    fig = plt.gcf()

    if true_params is None:
        plt.tight_layout()
        return fig, axs

    profile_result = ProfileResult(result, true_params)
    true_values = profile_result.get_true_parameter_values(scaled=True)
    param_names = profile_result.param_names

    plot_true_params_on_distribution(axs, true_values, param_names)
    plt.tight_layout()

    return fig, axs
