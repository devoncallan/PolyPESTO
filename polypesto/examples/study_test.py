from typing import List
from pathlib import Path

import numpy as np

from polypesto.core import Study, create_study_conditions
from polypesto.core import ParameterGroup
from polypesto.models.CRP2 import BinaryIrreversible

STUDY_DIR = Path(__file__).parent / "study"


def study_workflow():

    model = BinaryIrreversible(
        observables=["xA", "xB", "fA", "fB"],
    )

    true_params = ParameterGroup.create_parameter_grid(
        {
            "rA": [0.5, 1.0, 2.0],
            "rB": [1.0, 2.0, 4.0],
        },
        filter_fn=lambda p: p["rB"] > p["rA"],
    )

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

    study = Study.create(
        study_dir=STUDY_DIR,
        model=model,
        true_params=true_params,
        sim_conds=conds_dict,
        overwrite=True,
    )
    study = Study.load(STUDY_DIR, model)

    study.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10_000, n_chains=3),
        ),
        overwrite=True,
    )


def main():
    study_workflow()


if __name__ == "__main__":
    main()
