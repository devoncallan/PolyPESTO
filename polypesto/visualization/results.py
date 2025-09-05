from typing import Optional
import matplotlib.pyplot as plt

from polypesto.core.results import Result
from polypesto.core.problem import Problem

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


def plot_results(result: Result, problem: Problem, true_params: Optional[dict] = None):

    plot_all_measurements(problem.petab_problem.measurement_df)
    plt.gcf().savefig(problem.paths.measurements_fig, dpi=300)

    has_optimize_results = (
        len(result.optimize_result.list) > 0 if result.optimize_result else False
    )
    has_sample_results = result.sample_result 

    if result.optimize_result.list:
        plot_optimization_scatter(result, true_params)
        plt.gcf().savefig(problem.paths.optimization_scatter_fig, dpi=300)

        plot_waterfall(result)
        plt.gcf().savefig(problem.paths.waterfall_fig, dpi=300)

    if result.sample_result:
        plot_sampling_scatter(result, true_params)
        plt.gcf().savefig(problem.paths.sampling_scatter_fig, dpi=300)

        plot_confidence_intervals(result, true_params)
        plt.gcf().savefig(problem.paths.confidence_intervals_fig, dpi=300)

        plot_parameter_traces(result, true_params)
        plt.gcf().savefig(problem.paths.sampling_trace_fig, dpi=300)

    if result.profile_result:
        print("Plotting profiles...")
        print(result.profile_result)
        plot_profiles(result, true_params)
        plt.gcf().savefig(problem.paths.profile_fig, dpi=300)
