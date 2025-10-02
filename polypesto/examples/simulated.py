from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polypesto.core import (
    run_parameter_estimation,
    create_sim_conditions,
    simulate_problem,
)
from polypesto.core.pypesto import calculate_cis, create_ensemble, predict_with_ensemble
from polypesto.visualization import plot_results, plot_ensemble_predictions

# Model specific imports
from polypesto.models.binary import BinaryIrreversible
from polypesto.models.binary.utils import create_ensemble_pred_problem

from polypesto.examples.base import output_dirs

OUTPUT_DIR, ENSEMBLE_DIR = output_dirs(Path(__file__).stem)


def sim_workflow():

    # Initialize model with observables
    model = BinaryIrreversible(observables=["xA", "xB", "fA", "fB", "FA", "FB"])

    # Define true parameters and simulation conditions
    true_params = {"rA": 2.0, "rB": 1.0}
    sim_conds = create_sim_conditions(
        true_params=true_params,
        conds=dict(
            A0=[0.70, 0.50],
            B0=[0.30, 0.50],
        ),
        t_evals=np.arange(0.05, 0.61, 0.05),
        noise_levels=0.05,
    )

    # Simulate problem and create parameter estimation problem
    problem = simulate_problem(
        prob_dir=OUTPUT_DIR,
        model=model,
        conds=sim_conds,
        overwrite=True,
    )

    # Run parameter estimation (optimization + sampling)
    result = run_parameter_estimation(
        problem,
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=True,
    )
    calculate_cis(result, ci_level=0.95)

    # Visualize results
    plot_results(result, problem, true_params)

    # Predict using parameter ensemble from sampling
    pred_prob = create_ensemble_pred_problem(data_dir=ENSEMBLE_DIR, model=model)

    ensemble = create_ensemble(problem.pypesto_problem, result)
    ensemble_pred = predict_with_ensemble(
        ensemble, pred_prob.pypesto_problem, output_type="y"
    )

    plot_ensemble_predictions(ensemble_pred, problem)
    plt.gcf().savefig(problem.paths.ensemble_predictions_fig, dpi=300)


if __name__ == "__main__":

    sim_workflow()
