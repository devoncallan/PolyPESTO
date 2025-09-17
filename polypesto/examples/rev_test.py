from typing import List
from pathlib import Path

import numpy as np

from polypesto.visualization import plot_results

from polypesto.core import (
    run_parameter_estimation,
    create_sim_conditions,
    simulate_problem,
    pet,
    calculate_cis,
)
from polypesto.models.CRP2 import BinaryReversible

DATA_DIR = Path(__file__).parent / "polypesto/rev_test"


def sim_workflow():

    model = BinaryReversible(
        observables=["xA", "xB", "fA", "fB"],
    )
    model.fit_params["KAA"].set(
        scale=pet.C.LOG10, bounds=(1e-2, 1e2), estimate=True, nominal_value=0.1
    )

    true_params = {"rA": 1.0, "rB": 2.0, "KAA": 0.5}
    sim_conds = create_sim_conditions(
        true_params=true_params,
        conds=dict(
            A0=[0.30, 0.50, 0.70, 1.0],
            B0=[0.70, 0.50, 0.30, 1.0],
        ),
        t_evals=np.linspace(0, 0.50, 10),
        noise_levels=0.05,
    )

    problem = simulate_problem(
        prob_dir=DATA_DIR,
        model=model,
        conds=sim_conds,
    )

    result = run_parameter_estimation(
        problem,
        config=dict(
            optimize=dict(n_starts=10, method="Nelder-Mead"),
            sample=dict(n_samples=1000, n_chains=3),
        ),
        overwrite=True,
    )
    plot_results(result, problem, true_params)
    cis = calculate_cis(result, ci_level=0.95)


def main():
    sim_workflow()


if __name__ == "__main__":
    main()
