import argparse
from pathlib import Path

import numpy as np

from polypesto.core import ParameterGroup
from polypesto.core.study import Study, create_study_conditions
from polypesto.models.binary import BinaryIrreversible


def main(output_dir: str, overwrite: bool = False):

    model = BinaryIrreversible(observables=["FA", "FB"], obs_noise=0.01)

    true_params = ParameterGroup.create_parameter_grid(
        {
            "rA": [0.1, 0.5, 1.0, 2.0, 10.0],
            "rB": [0.1, 0.5, 1.0, 2.0, 10.0],
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
        study_dir=output_dir,
        model=model,
        true_params=true_params,
        sim_conds=conds_dict,
        overwrite=overwrite,
    )
    study = Study.load(output_dir, model)

    study.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10_000, n_chains=3),
        ),
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parameter estimation study for binary polymerization model"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="test_run",
        help="Name of the simulation run (default: test_run)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists",
    )
    args = parser.parse_args()

    # This file should be located in jobs/scripts
    JOBS_DIR = Path(__file__).parent.parent.absolute()
    OUTPUTS_DIR = JOBS_DIR / "outputs" / Path(__file__).stem
    STUDY_DIR = OUTPUTS_DIR / args.run_name
    overwrite = args.overwrite

    main(STUDY_DIR, overwrite=overwrite)
