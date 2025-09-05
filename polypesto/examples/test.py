from typing import List
from pathlib import Path

import numpy as np

from polypesto.visualization import plot_results
from polypesto.utils._patches import apply


from polypesto.core import Dataset, Experiment, Problem, run_parameter_estimation
from polypesto.core import create_sim_conditions, simulate_experiments
from polypesto.models.CRP2 import IrreversibleCPE

RAW_DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR = Path(__file__).parent / "data"


def modify_experiments(experiments: List[Experiment]) -> List[Experiment]:

    # Convert tkey in data to conversion
    # Add fA and fB to obs

    new_exps = []
    for exp in experiments:

        datasets = exp.data
        cond = exp.conds.values.to_dict()

        A0 = B0 = None
        if "A0" in cond and "B0" in cond:
            A0 = cond["A0"]
            B0 = cond["B0"]
        elif "fA0" in cond and "cM0" in cond:
            fA0 = cond["fA0"]
            cM0 = cond["cM0"]
            A0 = fA0 * cM0
            B0 = (1 - fA0) * cM0
        else:
            raise ValueError(
                f'Conditions must include either ("A0", "B0") or ("fA0", "cM0"). Actual: {list(cond.keys())}'
            )

        assert A0 is not None and B0 is not None
        assert A0 != 0 and B0 != 0

        fA0 = A0 / (A0 + B0)
        fB0 = B0 / (A0 + B0)

        new_datasets = []
        for ds in datasets:

            assert ds.tkey in ds.data.columns
            assert "xA" in ds.obs_map and "xB" in ds.obs_map

            ds.data[ds.tkey] = (
                fA0 * ds.data[ds.obs_map["xA"]] + fB0 * ds.data[ds.obs_map["xB"]]
            )

            mon_A = fA0 * (1 - ds.data[ds.obs_map["xA"]])
            mon_B = fB0 * (1 - ds.data[ds.obs_map["xB"]])
            ds.data["fA"] = mon_A / (mon_A + mon_B)
            ds.data["fB"] = mon_B / (mon_A + mon_B)

            ds.obs_map["fA"] = "fA"
            ds.obs_map["fB"] = "fB"

            new_datasets.append(ds)

        exp = Experiment(
            id=exp.id,
            conds=exp.conds,
            data=new_datasets,
        )
        new_exps.append(exp)

    return new_exps


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
