import numpy as np
import matplotlib.pyplot as plt
import pypesto
from matplotlib.axes import Axes
from typing import Dict, List, Tuple

from polypesto.core.study import SimulatedExperimentDict, ResultsDict, Study


def plot_comparisons(
    experiments: SimulatedExperimentDict, results: ResultsDict, axes: List[Axes]
) -> None:

    param_names = ["rA", "rB"]
    # param_flag = False

    for j, ((cond_id, p_id), experiment) in enumerate(experiments.items()):
        result = results[(cond_id, p_id)]

        lbs, ubs = pypesto.sample.calculate_ci_mcmc_sample(result)
        param_cis = [(10**lb, 10**ub) for (lb, ub) in zip(lbs, ubs)]

        true_params = experiment.true_params.to_dict()
        param_flag = true_params is not None

        # get fA

        for i, p_name in enumerate(param_names):
            ax = axes[i]
            ci = param_cis[i]

            mid = (ci[0] + ci[1]) / 2
            lower = mid - ci[0]
            upper = ci[1] - mid

            ax.errorbar(
                j,
                mid,
                yerr=[[lower], [upper]],
                fmt="o",
                capsize=5,
            )

            if param_flag:
                ax.axhline(true_params[p_name], color="red", linestyle="--")


def plot_all_comparisons(study: Study) -> None:
    parameter_ids = study.get_parameter_ids()

    for i, p_id in enumerate(parameter_ids):
        experiments = study.get_experiments(p_id)
        results = study.get_results(p_id)

        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.set_xlim(0, 1)

        plot_comparisons(experiments, results, axes)
        fig.savefig(f"comparison_plot_{p_id}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
