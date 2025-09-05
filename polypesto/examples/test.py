from pathlib import Path    

import numpy as np

from polypesto.visualization import plot_results
from polypesto.utils._patches import apply


from polypesto.core import Dataset, Experiment, Problem, run_parameter_estimation
from polypesto.core import create_sim_conditions
from polypesto.models.CRP2 import IrreversibleCPE

from polypesto.core.experiment import modify_experiments

# DATA_DIR = "/Users/devoncallan/Documents/GitHub/PolyPESTO/polypesto/examples/data"
RAW_DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR = Path(__file__).parent / "data"


def exp_workflow():

    model = IrreversibleCPE(
        observables=["xA", "xB", "fA", "fB"],
    )

    exp1 = Experiment.load(
        id="ELpMMA_3070",
        conds={"A0": 0.30, "B0": 0.70},
        data=[
            Dataset.load(
                path=RAW_DATA_DIR / "data_3060.csv",
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
                path=RAW_DATA_DIR / "data_5050.csv",
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

    from polypesto.core.simulate import simulate_experiments

    model = IrreversibleCPE(
        observables=["xA", "xB", "fA", "fB"],
    )

    true_params = {"rA": 1.0, "rB": 2.0}
    sim_conds = create_sim_conditions(
        true_params=true_params,
        conds=dict(
            A0=[0.30, 0.50],
            B0=[0.70, 0.50],
        ),
        # exp_ids=["exp1", "exp2"],
        t_evals=[np.linspace(0, 0.95, 20)] * 2,
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
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=True,
    )
    plot_results(result, problem, true_params)


def main():
    apply()

    exp_workflow()
    sim_workflow()


if __name__ == "__main__":
    main()
