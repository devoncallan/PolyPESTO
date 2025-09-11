from typing import Optional

import matplotlib.pyplot as plt

from polypesto.core import Result, Problem
from polypesto.visualization import (
    plot_optimization_scatter,
    plot_sampling_scatter,
    plot_confidence_intervals,
    plot_waterfall,
    plot_parameter_traces,
    plot_profiles,
    plot_ensemble_predictions,
    plot_all_measurements,
)
from pypesto.visualize import model_fit


def plot_results(result: Result, problem: Problem, true_params: Optional[dict] = None):

    plot_all_measurements(problem.petab_problem.measurement_df)
    plt.gcf().savefig(problem.paths.measurements_fig, dpi=300)

    has_optimize_results = (
        len(result.optimize_result.list) > 0 if result.optimize_result else False
    )
    has_sample_results = result.sample_result
    has_profile_results = (
        len(result.profile_result.list) > 0 if result.profile_result else False
    )

    if has_optimize_results:
        plot_optimization_scatter(result, true_params)
        plt.gcf().savefig(problem.paths.optimization_scatter_fig, dpi=300)

        plot_waterfall(result)
        plt.gcf().savefig(problem.paths.waterfall_fig, dpi=300)

    if has_sample_results:
        plot_sampling_scatter(result, true_params)
        plt.gcf().savefig(problem.paths.sampling_scatter_fig, dpi=300)

        plot_confidence_intervals(result, true_params)
        plt.gcf().savefig(problem.paths.confidence_intervals_fig, dpi=300)

        plot_parameter_traces(result, true_params)
        plt.gcf().savefig(problem.paths.sampling_trace_fig, dpi=300)

    if has_profile_results:
        print("Plotting profiles...")
        print(result.profile_result)
        plot_profiles(result, true_params)
        plt.gcf().savefig(problem.paths.profile_fig, dpi=300)

    ax = model_fit.visualize_optimized_model_fit(
        petab_problem=problem.petab_problem,
        result=result,
        pypesto_problem=problem.pypesto_problem,
    )
    # model_fit.
    plt.gcf().savefig(problem.paths.model_fit_fig, dpi=300)

    plt.close("all")
