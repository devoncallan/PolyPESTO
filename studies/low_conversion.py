"""Low-conversion sweep: for each true (rA, rB), 9 feed compositions at 5% conversion."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polypesto.core import ParameterGroup
from polypesto.core.study import Study, create_study_conditions
from polypesto.models.binary import BinaryIrreversible

from helpers.metrics import recovery_metrics, study_summary
from helpers.plots import plot_composition_vs_feed, plot_rA_rB_posteriors


HERE = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = HERE / "output" / "low_conversion"


def main(output_dir: Path, overwrite: bool = False):

    model = BinaryIrreversible(observables=["FA"])

    true_params = ParameterGroup.from_dict({
        "asym_A": {"rA": 2.0, "rB": 0.5},
        "neut":   {"rA": 1.0, "rB": 1.0},
        "asym_B": {"rA": 0.5, "rB": 2.0},
    })

    # Single problem: 9 feed compositions, all at 5% conversion (time==x in this model).
    fA = np.round(np.arange(0.1, 0.91, 0.1), 2)
    sim_conds = create_study_conditions(
        conds=dict(A0=[list(fA)], B0=[list(1 - fA)]),
        t_evals=np.array([0.05]),
        meas_noise=0.02,
    )

    study = Study.create(
        study_dir=output_dir,
        model=model,
        true_params=true_params,
        sim_conds=sim_conds,
        overwrite=overwrite,
    )
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
    agg = study_summary(metrics)
    print("\n==== Aggregate ====")
    print(agg.to_string())
    metrics.to_csv(output_dir / "recovery_metrics.csv")

    fig1, _ = plot_rA_rB_posteriors(study, level=0.95, show_nonconverged=True)
    fig1.savefig(output_dir / "posteriors_rA_rB.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, _ = plot_composition_vs_feed(study)
    fig2.savefig(output_dir / "composition_vs_feed.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.output_dir, overwrite=args.overwrite)
