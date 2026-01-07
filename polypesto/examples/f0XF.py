from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from polypesto.core import Dataset, Experiment, Problem
from polypesto.core.pypesto import calculate_cis, create_ensemble, predict_with_ensemble
from polypesto.examples.base import DATA_DIR, output_dirs

# Model specific imports
from polypesto.models.binary import BinaryIrreversible

# from polypesto.models.binary.utils import (
#     create_ensemble_pred_problem,
#     modify_experiments,
# )

OUTPUT_DIR = output_dirs(Path(__file__).stem)


def main():

    # Initialize model with observables
    model = BinaryIrreversible(observables=["FA"])

    df = pd.read_csv(DATA_DIR / "f0XF_test.csv")
    # df = pd.read_csv(DATA_DIR / "f0XF_APSA_Vim.csv")
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
                    noise_map={"FA": 0.01},
                    # noise_map={"FA": 0.03},
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
            sample=dict(n_samples=10_000, n_chains=3),
        ),
        overwrite=True,
    )

    calculate_cis(result, ci_level=0.95)


def main2():

    model = BinaryIrreversible(observables=["FA"])

    df_exp = pd.read_csv(DATA_DIR / "f0XF_excel.csv")
    # unique_conds = [float(c) for c in df_exp["fA0"].unique()]
    unique_conds = [0.4]

    exps: List[Experiment] = []
    for cond in unique_conds:
        df = df_exp[df_exp["fA0"] == cond]

        df["FA"] = (df["fA0"] - (1 - df["X"]) * df["fA"]) / df["X"]

        dFAdfA = ((1 - df["X"]) * df["dfA"] / df["X"]) ** 2
        dFAdX = ((df["fA"] - df["fA0"]) * df["dX"] / df["X"] ** 2) ** 2
        df["dFA"] = (dFAdfA + dFAdX) ** 0.5
        print(df["dFA"])

        # df["dFA"] = np.sqrt()

        exp = Experiment.load(
            id=f"fA0_{int(cond*100)}",
            conds={"A0": cond, "B0": 1 - cond},  # Define initial conditions
            data=[  # Load conversion data and map to observables
                Dataset.load(
                    df,
                    tkey="X",
                    obs_map={"FA": "FA"},
                    noise_map={"FA": 0.001}
                    # noise_map={"FA": "dFA"},
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
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=True,
    )

    calculate_cis(result, ci_level=0.95)


if __name__ == "__main__":
    main()
