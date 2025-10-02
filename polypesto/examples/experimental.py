from pathlib import Path

import matplotlib.pyplot as plt

from polypesto.core import Dataset, Experiment, Problem, run_parameter_estimation
from polypesto.core.pypesto import calculate_cis, create_ensemble, predict_with_ensemble
from polypesto.examples.base import DATA_DIR, output_dirs

# Model specific imports
from polypesto.models.binary import BinaryIrreversible
from polypesto.models.binary.utils import (
    create_ensemble_pred_problem,
    modify_experiments,
)
from polypesto.visualization import plot_ensemble_predictions, plot_results

OUTPUT_DIR, ENSEMBLE_DIR = output_dirs(Path(__file__).stem)


def main():

    # Initialize model with observables
    model = BinaryIrreversible(observables=["xA", "xB", "fA", "fB"])

    # Load experimental data from `data/` directory
    exp1 = Experiment.load(
        id="ELpMMA_3070",
        conds={"A0": 0.30, "B0": 0.70},  # Define initial conditions
        data=[  # Load conversion data and map to observables
            Dataset.load(
                DATA_DIR / "data_3060.csv",
                tkey="Time[min]",
                obs_map={"xA": "Conversion ELp", "xB": "Conversion MMA"},
            )
        ],
    )

    exp2 = Experiment.load(
        id="ELpMMA_5050",
        conds={"A0": 0.50, "B0": 0.50},  # Define initial conditions
        data=[  # Load conversion data and map to observables
            Dataset.load(
                DATA_DIR / "data_5050.csv",
                tkey="Time[min]",
                obs_map={"xA": "Conversion ELp", "xB": "Conversion MMA"},
            )
        ],
    )

    # Format experiments for parameter estimation
    exps = [exp1, exp2]
    exps = modify_experiments(exps)

    # Create parameter estimation problem from experiments
    problem = Problem.from_experiments(
        output_dir=OUTPUT_DIR,
        model=model,
        experiments=exps,
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
    plot_results(result, problem)

    # Predict using parameter ensemble from sampling
    pred_prob = create_ensemble_pred_problem(problem.paths.ensemble_dir, model=model)

    ensemble = create_ensemble(problem.pypesto_problem, result)
    ensemble_pred = predict_with_ensemble(
        ensemble, pred_prob.pypesto_problem, output_type="y"
    )

    plot_ensemble_predictions(ensemble_pred, problem)
    plt.gcf().savefig(problem.paths.ensemble_predictions_fig, dpi=300)


if __name__ == "__main__":
    main()
