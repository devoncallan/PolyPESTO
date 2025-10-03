from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from polypesto.core import Dataset, Experiment, Problem
from polypesto.core.pypesto import calculate_cis, create_ensemble, predict_with_ensemble
from polypesto.examples.base import DATA_DIR, output_dirs

# Model specific imports
from polypesto.models.binary import BinaryIrreversible
from polypesto.models.binary.utils import (
    create_ensemble_pred_problem,
    modify_experiments,
)

OUTPUT_DIR = output_dirs(Path(__file__).stem)


def main():

    # Initialize model with observables
    model = BinaryIrreversible(observables=["FA"])

    df = pd.read_csv(DATA_DIR / "f0XF_APSA_Vim.csv")
    unique_conds = [float(c) for c in df["fA0"].unique()]

    exps = []
    for cond in unique_conds:
        df_exp = df[df["fA0"] == cond]
        # if cond == 0.05 or cond == 0.10:
        #     continue

        exp = Experiment.load(
            id=f"fA0_{int(cond*100)}",
            conds={"A0": cond, "B0": 1 - cond},  # Define initial conditions
            data=[  # Load conversion data and map to observables
                Dataset.load(
                    df_exp,
                    tkey="X",
                    obs_map={"FA": "FA"},
                    noise_map={"FA": 0.03},
                )
            ],
        )
        exps.append(exp)

    # Create parameter estimation problem from experiments
    problem = Problem.from_experiments(
        output_dir=OUTPUT_DIR,
        model=model,
        experiments=exps,
    )

    # Run parameter estimation (optimization + sampling)
    result = problem.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=1000, n_chains=3),
        ),
        overwrite=True,
    )

    calculate_cis(result, ci_level=0.95)


if __name__ == "__main__":
    main()
