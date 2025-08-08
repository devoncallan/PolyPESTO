import matplotlib.pyplot as plt

from polypesto.core.experiment import Experiment, SimulatedExperiment
from polypesto.core.results import Result
from polypesto.core.study import Study
from polypesto.visualization import (
    plot_optimization_scatter,
    plot_sampling_scatter,
    plot_confidence_intervals,
    plot_waterfall,
    plot_parameter_traces,
    plot_profiles,
    plot_ensemble_predictions,
    plot_all_measurements,
    # plot_all_comparisons,
)


def plot_results(exp: Experiment | SimulatedExperiment, result: Result):

    true_params = None
    if isinstance(exp, SimulatedExperiment):
        true_params = exp.true_params.to_dict()
        exp = exp.experiment

    axs = plot_all_measurements(exp.petab_problem.measurement_df)
    plt.gcf().savefig(exp.paths.measurements_data_plot, dpi=300)

    fig, ax = plot_optimization_scatter(result, true_params)
    plt.gcf().savefig(exp.paths.optimization_scatter_plot, dpi=300)

    fig, ax = plot_waterfall(result)
    plt.gcf().savefig(exp.paths.waterfall_plot, dpi=300)

    fig, ax = plot_sampling_scatter(result, true_params)
    plt.gcf().savefig(exp.paths.sampling_scatter_plot, dpi=300)

    fig, ax = plot_parameter_traces(result, true_params)
    plt.gcf().savefig(exp.paths.sampling_trace_plot, dpi=300)

    fig, ax = plot_profiles(result, true_params)
    plt.gcf().savefig(exp.paths.profile_plot, dpi=300)

    fig, ax = plot_confidence_intervals(result, true_params)
    plt.gcf().savefig(exp.paths.confidence_intervals_plot, dpi=300)

    plt.close("all")


def plot_all_results(
    study: Study,
):

    for (cond_id, p_id), result in study.results.items():

        exp = study.experiments[(cond_id, p_id)]
        true_params = exp.true_params.to_dict()

        plot_results(exp, result)


def plot_all_ensemble_predictions(study: Study, test_study: Study):
    """
    Plot all ensemble predictions for the given study and test study.
    """

    from polypesto.core.pypesto import create_ensemble, predict_with_ensemble

    for (cond_id, p_id), result in study.results.items():

        exp = study.experiments[(cond_id, p_id)]

        ensemble = create_ensemble(exp, result)

        test_exp = list(test_study.experiments.values())[0]
        ensemble_pred = predict_with_ensemble(ensemble, test_exp, output_type="y")

        fig, ax = plot_ensemble_predictions(ensemble_pred, test_exp)
        plt.gcf().savefig(exp.paths.ensemble_predictions_plot, dpi=300)
        plt.close("all")
