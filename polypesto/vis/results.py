from typing import Dict, Optional

from polypesto.core import Result, Problem
from polypesto.core.pypesto import (
    has_optimization_results,
    has_profile_results,
    has_sampling_results,
    has_problem_results,
)
from polypesto.vis import (
    plot_optimization_scatter,
    plot_sampling_scatter,
    plot_confidence_intervals,
    plot_waterfall,
    plot_parameter_traces,
    plot_profiles,
    plot_optimized_model_fit,
    plot_ensemble_predictions,
    plot_all_measurements,
)
from .base import save_plot


def plot_results(
    result: Result, problem: Problem, true_params: Optional[Dict[str, float]] = None
) -> None:
    """Plots the results of the parameter estimation.

    Args:
        result (Result): The result object containing the optimization results.
        problem (Problem): The problem object containing the problem definition.
        true_params (Optional[dict], optional): The true parameter values. Defaults to None.
    """

    if has_problem_results(result):

        with save_plot(problem.paths.measurements_fig):
            plot_all_measurements(problem.petab_problem.measurement_df)

    if has_optimization_results(result):

        with save_plot(problem.paths.optimization_scatter_fig):
            plot_optimization_scatter(result, true_params)

        with save_plot(problem.paths.waterfall_fig):
            plot_waterfall(result)

        with save_plot(problem.paths.model_fit_fig):
            plot_optimized_model_fit(result, problem)

    if has_sampling_results(result):

        with save_plot(problem.paths.sampling_scatter_fig):
            plot_sampling_scatter(result, true_params)

        with save_plot(problem.paths.confidence_intervals_fig):
            plot_confidence_intervals(result, true_params)

        with save_plot(problem.paths.sampling_trace_fig):
            plot_parameter_traces(result, true_params)

    if has_profile_results(result):

        with save_plot(problem.paths.profile_fig):
            plot_profiles(result, true_params)
