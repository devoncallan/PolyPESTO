"""Conversion-mismatch test: data generated at 5% conversion, fit assuming 3%.

Same 9-feed-composition design as low_conversion.py. After simulation, each
problem's measurements.tsv has its ``time`` column rewritten from 0.05 to 0.03
so the fitter treats the 5%-conversion data as if it were sampled at 3%.

The bias of the recovered (rA, rB) from truth quantifies how sensitive
reactivity-ratio extraction is to conversion misestimation.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polypesto.core import ParameterGroup
from polypesto.core.study import Study, create_study_conditions
from polypesto.models.binary import BinaryIrreversible

from helpers.metrics import recovery_metrics, study_summary
from helpers.plots import plot_rA_rB_posteriors


HERE = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = HERE / "output" / "conversion_mismatch"

TRUE_X = 0.05
ASSUMED_X = 0.03


def _relabel_time(study_dir: Path, new_time: float) -> None:
    """Rewrite the ``time`` column of every measurements.tsv under ``study_dir``."""
    for meas_path in Path(study_dir).rglob("measurements.tsv"):
        df = pd.read_csv(meas_path, sep="\t")
        df["time"] = new_time
        df.to_csv(meas_path, sep="\t", index=False)


def main(output_dir: Path, overwrite: bool = False):

    model = BinaryIrreversible(observables=["FA"])

    true_params = ParameterGroup.from_dict({
        "r10_1":   {"rA": 10.0, "rB": 1.0},
        "r1_10":   {"rA": 1.0,  "rB": 10.0},
        "r10_0.1": {"rA": 10.0, "rB": 0.1},
        "r0.1_10": {"rA": 0.1,  "rB": 10.0},
    })

    fA = np.round(np.arange(0.1, 0.91, 0.1), 2)
    sim_conds = create_study_conditions(
        conds=dict(A0=[list(fA)], B0=[list(1 - fA)]),
        t_evals=np.array([TRUE_X]),
        meas_noise=0.02,
    )

    Study.create(
        study_dir=output_dir,
        model=model,
        true_params=true_params,
        sim_conds=sim_conds,
        overwrite=overwrite,
    )
    _relabel_time(output_dir, ASSUMED_X)
    study = Study.load(output_dir, model)

    study.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=25, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=overwrite,
    )

    summary = study.results_summary()
    metrics = recovery_metrics(summary, study=study)
    print(f"\n==== true_x={TRUE_X}, assumed_x={ASSUMED_X} ====")
    print(study_summary(metrics).to_string())
    metrics.to_csv(output_dir / "recovery_metrics.csv")

    fig, _ = plot_rA_rB_posteriors(study, level=0.95, show_nonconverged=True)
    fig.savefig(output_dir / "posteriors_rA_rB.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.output_dir, overwrite=args.overwrite)
