from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


import pypesto.visualize as vis
from pypesto.ensemble import EnsemblePrediction

from polypesto.core.problem import Problem

########################
### Prediction Plots ###
########################


def plot_ensemble_predictions(
    ensemble_pred: EnsemblePrediction, prob: Problem, levels=[90, 95, 99], **kwargs
) -> Tuple[Figure, Axes]:

    mdf = prob.petab_problem.measurement_df
    mdf["conditionId"] = mdf["simulationConditionId"]

    axs = vis.sampling_prediction_trajectories(
        ensemble_prediction=ensemble_pred,
        levels=levels,
        measurement_df=mdf,
        groupby="condition",
        **kwargs,
    )

    fig = plt.gcf()
    plt.tight_layout()

    return fig, axs
