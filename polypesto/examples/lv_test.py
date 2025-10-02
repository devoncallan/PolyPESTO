from pathlib import Path

import numpy as np

from polypesto.core import calculate_cis, create_sim_conditions, simulate_problem
from polypesto.examples.base import output_dirs

# Model specific imports
from polypesto.models.example.lotka_volterra import LotkaVolterra

OUTPUT_DIR, ENSEMBLE_DIR = output_dirs(Path(__file__).stem)


def main():

    # Initialize model with default observables
    model = LotkaVolterra()

    # Define true parameters and simulation conditions
    true_params = {"a": 1.1, "b": 0.4, "c": 0.4, "d": 0.1}
    sim_conds = create_sim_conditions(
        true_params=true_params,
        conds=dict(x=[4.8], y=[2.2]),
        t_evals=np.linspace(0, 10, 200),
        noise_levels=0.02,
    )

    problem = simulate_problem(
        prob_dir=OUTPUT_DIR,
        model=model,
        conds=sim_conds,
    )

    result = problem.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=200, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=True,
    )
    calculate_cis(result, ci_level=0.95)


if __name__ == "__main__":
    main()
