import matplotlib.pyplot as plt

from polypesto.core.study import Study
from .results import plot_results


def plot_all_results(study: Study):

    for key, result in study.results.items():

        true_params = study.true_params.by_id(key[1]).to_dict()
        plot_results(study.results[key], study.problems[key], true_params)


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
