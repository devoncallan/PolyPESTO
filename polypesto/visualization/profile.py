from typing import Optional, Dict

import matplotlib.pyplot as plt

import pypesto.visualize as vis
from pypesto.result import Result

# Import our result handlers
from polypesto.core.results import ProfileResult
from .true import plot_true_params_on_distribution

#######################
### Profiling Plots ###
#######################


def plot_profiles(
    result: Result, true_params: Optional[Dict[str, float]] = None, **kwargs
):

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
