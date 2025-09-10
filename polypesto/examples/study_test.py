from typing import List
from pathlib import Path

import numpy as np

from polypesto.visualization import plot_results
from polypesto.utils._patches import apply


from polypesto.core import Dataset, Experiment, Problem, run_parameter_estimation
from polypesto.core import create_sim_conditions, simulate_experiments
from polypesto.core import ParameterGroup
from polypesto.models.CRP2 import IrreversibleCPE

STUDY_DIR = Path(__file__).parent / "study"


def study_workflow():

    model = IrreversibleCPE(
        observables=["xA", "xB", "fA", "fB"],
    )

    true_params = ParameterGroup.create_parameter_grid(
        {
            "rA": [0.5, 1.0, 2.0],
            "rB": [1.0, 2.0, 4.0],
        },
        filter_fn=lambda p: p["rB"] > p["rA"],
    )
    print(true_params.to_dict())

    from polypesto.core.study import Study, create_study_conditions

    conds_dict = create_study_conditions(
        conds=dict(
            A0=[[0.25, 0.50], [0.50, 0.75], [0.25, 0.75]],
            B0=[[0.75, 0.50], [0.50, 0.25], [0.75, 0.25]],
        ),
        t_evals=np.linspace(0, 0.95, 20),
        noise_levels=0.02,
    )
    for key, conds in conds_dict.items():
        print(f"Conditions for problem {key}:")
        for sim_cond in conds:
            print(f"{key}: {sim_cond.exp_id}, {sim_cond.values.to_dict()}")

    # study = Study.load(STUDY_DIR, model)
    # study = Study.create(
    #     study_dir=STUDY_DIR,
    #     model=model,
    #     true_params=true_params,
    #     sim_conds=conds_dict,
    #     overwrite=True,
    # )
    study = Study.load(STUDY_DIR, model)

    from polypesto.visualization.results import plot_results

    for (cond_id, p_id), result in study.results.items():

        problem = study.problems[(cond_id, p_id)]
        true_params = study.true_params.by_id(p_id).to_dict()
        plot_results(result, problem, true_params)

    # study.run_parameter_estimation(
    #     config=dict(
    #         optimize=dict(n_starts=50, method="Nelder-Mead"),
    #         sample=dict(n_samples=10_000, n_chains=3),
    #     ),
    #     overwrite=True,
    # )
    # print(study.true_params)
    # print(study.sim_params)
    # # print(study.problems)

    # for (prob_id, param_id), problem in study.problems.items():
    #     print(f"Problem ID: {prob_id}, Parameter ID: {param_id}")
    # print(problem.experiments)
    # true_params = {"rA": 1.0, "rB": 2.0}
    # sim_conds = create_sim_conditions(
    #     true_params=true_params,
    #     conds=dict(
    #         A0=[0.30, 0.50],
    #         B0=[0.70, 0.50],
    #     ),
    #     # exp_ids=["exp1", "exp2"],
    #     t_evals=np.linspace(0, 0.95, 20),
    #     noise_levels=0.02,
    # )

    # problem = simulate_experiments(
    #     data_dir=DATA_DIR,
    #     model=model,
    #     conds=sim_conds,
    # )

    # result = run_parameter_estimation(
    #     problem,
    #     config=dict(
    #         optimize=dict(n_starts=50, method="Nelder-Mead"),
    #         sample=dict(n_samples=10000, n_chains=3),
    #     ),
    #     overwrite=True,
    # )
    # plot_results(result, problem, true_params)


def main():
    apply()

    study_workflow()


if __name__ == "__main__":
    main()
