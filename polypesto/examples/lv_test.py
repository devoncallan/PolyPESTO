from typing import List
from pathlib import Path

import numpy as np

from polypesto.visualization import plot_results
from polypesto.utils._patches import apply


from polypesto.core import Dataset, Experiment, Problem, run_parameter_estimation
from polypesto.core import create_sim_conditions, simulate_experiments
from polypesto.models.example.lotka_volterra_ode import LotkaVolterraODE

DATA_DIR = Path(__file__).parent / "data/lv_test"


def sim_workflow():

    model = LotkaVolterraODE()

    true_params = {"a": 1.1, "b": 0.4, "c": 0.4, "d": 0.1}
    sim_conds = create_sim_conditions(
        true_params=true_params,
        conds=dict(x=[4.8], y=[2.2]),
        t_evals=np.linspace(0, 10, 200),
        noise_levels=0.02,
    )

    problem = simulate_experiments(
        data_dir=DATA_DIR,
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


def main():
    apply()
    sim_workflow()


if __name__ == "__main__":
    main()
