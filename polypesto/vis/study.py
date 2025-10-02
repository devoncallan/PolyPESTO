import matplotlib.pyplot as plt

from polypesto.core.study import Study
from .results import plot_results, plot_ensemble_predictions


def plot_all_results(study: Study):

    for key, result in study.results.items():

        _, param_id = key
        true_params = study.true_params.by_id(param_id).to_dict()
        plot_results(result, study.problems[key], true_params)


def plot_all_ensemble_predictions(study: Study, test_study: Study):
    pass
