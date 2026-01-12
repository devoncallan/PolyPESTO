from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd

import pypesto.visualize as vis
from pypesto.visualize import model_fit
from pypesto.result import Result
from pypesto.C import RDATAS  # type: ignore
from amici.petab.simulations import rdatas_to_measurement_df  # type: ignore

from polypesto.core.problem import Problem
from polypesto.core.pypesto import has_optimization_results, get_true_param_values
from .true import plot_true_params_on_pairgrid

##########################
### Optimization Plots ###
##########################


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

    # Return if no grid axes or parameter names
    if not hasattr(grid, "axes") or len(grid.axes) == 0:
        plt.tight_layout()
        return fig, grid

    plot_true_params_on_pairgrid(grid, true_values)

    plt.tight_layout()

    return fig, grid


def plot_optimized_model_fit(
    problem: Problem,
    result: Result,
    overwrite: bool = False,
    write_measurement_df: bool = False,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plots the model fit after optimization.

    Args:
        problem (Problem): The problem object containing the problem definition.
        result (Result): The result object containing the optimization results.
        overwrite (bool): Whether to overwrite an existing predictions file.
        write_measurement_df (bool): If True, also write the predicted measurements
            (PEtab-style measurement dataframe) next to results.hdf5.

    Returns:
        (Figure, Axes): The figure and axes objects.
    """

    if not has_optimization_results(result):
        return plt.subplots()

    fit_out = model_fit.visualize_optimized_model_fit(
        petab_problem=problem.petab_problem,
        result=result,
        pypesto_problem=problem.pypesto_problem,
        return_dict=True,
        **kwargs,
    )

    # Extract axes from return_dict payload
    ax = fit_out["axes"] if isinstance(fit_out, dict) else fit_out

    if write_measurement_df:
        # Convert AMICI rdatas back into a PEtab measurement-style dataframe and persist
        payloads = fit_out if isinstance(fit_out, list) else [fit_out]
        dfs = []
        for ix, payload in enumerate(payloads):
            rdatas = payload["objective_result"][RDATAS]
            df = rdatas_to_measurement_df(
                rdatas,
                problem.pypesto_problem.objective.amici_model,
                problem.petab_problem.measurement_df,
            )
            if len(payloads) > 1:
                df.insert(0, "model_index", ix)
            dfs.append(df)

        if dfs:
            pred_df = pd.concat(dfs, ignore_index=True)
            out_path = problem.paths.model_fit_measurements
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not out_path.exists():
                pred_df.to_csv(out_path, sep="\t", index=False)

    fig = plt.gcf()
    plt.tight_layout()

    return fig, ax
