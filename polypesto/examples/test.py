from typing import List
from pathlib import Path

import numpy as np

# from polypesto.models import ModelBase
from polypesto.visualization import plot_results
from polypesto.core import Dataset, Experiment, Problem, run_parameter_estimation
from polypesto.core import create_sim_conditions, simulate_problem
from polypesto.models.CRP2 import BinaryIrreversible


FILE = Path(__file__)
RAW_DATA_DIR = FILE.parent / FILE.stem / "raw"
DATA_DIR = FILE.parent / FILE.stem / "data"
ENS_DIR = FILE.parent / FILE.stem / "ensemble"


def exp_workflow():

    from polypesto.models.CRP2.utils import modify_experiments

    model = BinaryIrreversible(
        observables=["xA", "xB", "fA", "fB"],
    )

    exp1 = Experiment.load(
        id="ELpMMA_3070",
        conds={"A0": 0.30, "B0": 0.70},
        data=[
            Dataset.load(
                RAW_DATA_DIR / "data_3060.csv",
                tkey="Time[min]",
                obs_map={"xA": "Conversion ELp", "xB": "Conversion MMA"},
            )
        ],
    )

    exp2 = Experiment.load(
        id="ELpMMA_5050",
        conds={"A0": 0.50, "B0": 0.50},
        data=[
            Dataset.load(
                RAW_DATA_DIR / "data_5050.csv",
                tkey="Time[min]",
                obs_map={"xA": "Conversion ELp", "xB": "Conversion MMA"},
            )
        ],
    )

    exps = [exp1, exp2]
    exps = modify_experiments(exps)

    problem = Problem.from_experiments(
        data_dir=DATA_DIR,
        model=model,
        experiments=exps,
    )

    result = run_parameter_estimation(
        problem,
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=True,
    )
    plot_results(result, problem)


def sim_workflow():

    model = BinaryIrreversible(
        observables=["xA", "xB", "fA", "fB", "FA", "FB"],
        # observables=["xA", "fA", "FA"],
    )

    # true_params = {"rA": 1.0, "rB": 2.0}
    true_params = {"rA": 2.0, "rB": 1.0}
    sim_conds = create_sim_conditions(
        true_params=true_params,
        conds=dict(
            A0=[0.70, 0.50],
            B0=[0.30, 0.50],
            # A0=[0.30, 0.50],
            # B0=[0.70, 0.50],
        ),
        t_evals=np.arange(0.05, 0.61, 0.05),
        noise_levels=0.05,
    )

    problem = simulate_problem(
        prob_dir=DATA_DIR,
        model=model,
        conds=sim_conds,
        overwrite=False,
    )

    result = run_parameter_estimation(
        problem,
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=False,
    )
    plot_results(result, problem, true_params)

    import matplotlib.pyplot as plt
    from polypesto.models.CRP2.utils import create_ensemble_pred_problem
    from polypesto.core.pypesto import create_ensemble, predict_with_ensemble
    from polypesto.visualization.predict import plot_ensemble_predictions

    pred_prob = create_ensemble_pred_problem(data_dir=ENS_DIR, model=model)

    ensemble = create_ensemble(problem.pypesto_problem, result)
    ensemble_pred = predict_with_ensemble(
        ensemble, pred_prob.pypesto_problem, output_type="y"
    )

    plot_ensemble_predictions(ensemble_pred, problem)
    plt.gcf().savefig(problem.paths.ensemble_predictions_fig, dpi=300)


def main():

    # exp_workflow()
    sim_workflow()


if __name__ == "__main__":

    main()
