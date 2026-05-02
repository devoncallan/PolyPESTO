import shutil
from typing import Dict, Optional

from polypesto.core import Result, Problem
from polypesto.core.pypesto import (
    has_optimization_results,
    has_profile_results,
    has_sampling_results,
    has_problem_results,
    save_sampling_trace,
)
from polypesto.vis import (
    plot_optimization_scatter,
    plot_sampling_scatter,
    plot_confidence_intervals,
    plot_waterfall,
    plot_parameter_traces,
    plot_profiles,
    plot_optimized_model_fit,
    plot_all_measurements,
)
from .base import save_plot


def plot_results(
    result: Result,
    problem: Problem,
    true_params: Optional[Dict[str, float]] = None,
    overwrite: bool = False,
) -> None:
    """Plots the results of the parameter estimation.

    Args:
        result (Result): The result object containing the optimization results.
        problem (Problem): The problem object containing the problem definition.
        true_params (Optional[dict], optional): The true parameter values. Defaults to None.
    """

    if has_problem_results(result):

        if overwrite:
            fig_dir = problem.paths.figures_dir
            if fig_dir.exists():
                shutil.rmtree(fig_dir)
            fig_dir.mkdir(parents=True, exist_ok=True)

        with save_plot(problem.paths.measurements_fig, overwrite=overwrite):
            plot_all_measurements(problem.petab_problem.measurement_df)

    if has_optimization_results(result):

        with save_plot(problem.paths.optimization_scatter_fig, overwrite=overwrite):
            plot_optimization_scatter(result, true_params)

        with save_plot(problem.paths.waterfall_fig, overwrite=overwrite):
            plot_waterfall(result)

        with save_plot(problem.paths.model_fit_fig, overwrite=overwrite):
            plot_optimized_model_fit(
                problem,
                result,
                overwrite=overwrite,
                write_measurement_df=True,
            )

    if has_sampling_results(result):

        with save_plot(problem.paths.sampling_scatter_fig, overwrite=overwrite):
            # print("Plotting sampling scatter...")
            plot_sampling_scatter(result, true_params, show_bounds=True)
            # plot_sampling_scatter

        with save_plot(problem.paths.confidence_intervals_fig, overwrite=overwrite):
            plot_confidence_intervals(result, true_params)

        with save_plot(problem.paths.sampling_trace_fig, overwrite=overwrite):
            plot_parameter_traces(result, true_params)

        # Persist full sampling trace as a CSV next to results.hdf5
        save_sampling_trace(
            result,
            out_path=problem.paths.sampling_trace,
            overwrite=overwrite,
            exclude_burn_in=True,
            unscale_params=True,
            chain="all",
            wide=True,
        )

    if has_profile_results(result):

        with save_plot(problem.paths.profile_fig, overwrite=overwrite):
            plot_profiles(result, true_params)
