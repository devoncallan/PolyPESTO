from pathlib import Path

import numpy as np

from polypesto.core import ParameterGroup
from polypesto.core.study import Study, create_study_conditions
from polypesto.examples.base import output_dirs
from polypesto.models.binary import BinaryIrreversible

OUTPUT_DIR = output_dirs(Path(__file__).stem)


def main():

    model = BinaryIrreversible(observables=["FA", "FB"], obs_noise=0.01)

    true_params = ParameterGroup.create_parameter_grid(
        {
            "rA": [0.5],
            "rB": [1.0, 2.0],
        },
        filter_fn=lambda p: p["rB"] > p["rA"],
    )

    conds_dict = create_study_conditions(
        conds=dict(
            A0=[[0.1], [0.3], [0.50], [0.7], [0.9]],
            B0=[[0.9], [0.7], [0.50], [0.3], [0.1]],
        ),
        t_evals=np.linspace(0.05, 0.80, 16),
        meas_noise=0.01,
    )

    study = Study.create(
        study_dir=OUTPUT_DIR,
        model=model,
        true_params=true_params,
        sim_conds=conds_dict,
        overwrite=True,
    )
    study = Study.load(OUTPUT_DIR, model)

    study.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10_000, n_chains=3),
        ),
        overwrite=True,
    )


if __name__ == "__main__":
    main()
