from pathlib import Path

import numpy as np

from polypesto.visualization import plot_results
from polypesto.core import run_parameter_estimation, calculate_cis
from polypesto.core import create_sim_conditions, simulate_problem
from polypesto.models.example.lotka_volterra import LotkaVolterra

DATA_DIR = Path(__file__).parent / "polypesto/lv_test"


def sim_workflow():

    model = LotkaVolterra()

    true_params = {"a": 1.1, "b": 0.4, "c": 0.4, "d": 0.1}
    sim_conds = create_sim_conditions(
        true_params=true_params,
        conds=dict(x=[4.8], y=[2.2]),
        t_evals=np.linspace(0, 10, 200),
        noise_levels=0.02,
    )

    problem = simulate_problem(
        prob_dir=DATA_DIR,
        model=model,
        conds=sim_conds,
    )

    result = run_parameter_estimation(
        problem,
        config=dict(
            optimize=dict(n_starts=200, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=True,
    )
    plot_results(result, problem, true_params)
    cis = calculate_cis(result, ci_level=0.95)


def main():
    sim_workflow()


if __name__ == "__main__":
    main()
