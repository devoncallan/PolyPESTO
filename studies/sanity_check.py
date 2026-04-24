"""Minimal end-to-end sanity check for the Study workflow.

Simulates copolymerization data for a tiny grid of true (rA, rB) values across
a couple of feed compositions using BinaryIrreversible, fits with the same
model, and prints a recovery summary. Intended to be fast (~a few minutes).
"""

import argparse
from pathlib import Path

import numpy as np

from polypesto.core import ParameterGroup
from polypesto.core.study import Study, create_study_conditions
from polypesto.models.binary import BinaryIrreversible


HERE = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = HERE / "output" / "sanity_check"


def main(output_dir: Path, overwrite: bool = False):

    model = BinaryIrreversible(observables=["FA"])

    # Two true-parameter sets (one asymmetric, one "neutral").
    true_params = ParameterGroup.from_dict(
        {
            "p_asym": {"rA": 2.0, "rB": 0.5},
            "p_neut": {"rA": 1.0, "rB": 1.0},
        }
    )

    # Two feed compositions -> two prob_ids.
    sim_conds = create_study_conditions(
        conds=dict(
            A0=[[0.30], [0.70]],
            B0=[[0.70], [0.30]],
        ),
        t_evals=np.arange(0.05, 0.61, 0.10),
        meas_noise=0.01,
    )

    study = Study.create(
        study_dir=output_dir,
        model=model,
        true_params=true_params,
        sim_conds=sim_conds,
        overwrite=overwrite,
    )
    # Round-trip: confirm the study is also loadable from disk.
    study = Study.load(output_dir, model)

    study.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=10, method="Nelder-Mead"),
            sample=dict(n_samples=1000, n_chains=3),
        ),
        overwrite=overwrite,
    )

    summary = study.results_summary()
    print("\n\n==== Study results summary ====")
    print(summary)

    summary_path = Path(output_dir) / "results_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args.output_dir, overwrite=args.overwrite)
