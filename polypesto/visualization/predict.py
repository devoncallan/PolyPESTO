from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


import pypesto.visualize as vis
from pypesto.ensemble import EnsemblePrediction

from polypesto.core.experiment import Experiment
from .base import safe_plot

########################
### Prediction Plots ###
########################


@safe_plot
def plot_ensemble_predictions(
    ensemble_pred: EnsemblePrediction, exp: Experiment, levels=[90, 95, 99]
) -> Tuple[Figure, Axes]:

    mdf = exp.petab_problem.measurement_df
    mdf["conditionId"] = mdf["simulationConditionId"]

    axs = vis.sampling_prediction_trajectories(
        ensemble_prediction=ensemble_pred,
        levels=levels,
        measurement_df=mdf,
        size=(10, 10),
    )

    fig = plt.gcf()
    plt.tight_layout()

    return fig, axs
