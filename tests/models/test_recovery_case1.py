"""
Parameter-estimation recovery test on BinaryReversible1TimeFit (Case I).

4 free params (rA, rB, KAA_Tref, dH_R) plus Van't Hoff T-dependence. We
spread simulation conditions across multiple temperatures so KAA_Tref and
dH_R are identifiable from the data (a single-temperature experiment can't
disentangle them).

Marked `slow`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polypesto.models.binary import BinaryReversible1TimeFit

from .conftest import assert_recovered, run_recovery


pytestmark = pytest.mark.slow


def test_case1_recovery(tmp_path: Path):
    # Truth: moderate rA/rB asymmetry, KAA_Tref and dH_R near Tsarevsky NLS
    # point estimate so the priors aren't fighting the data.
    true_params = {
        "rA": 1.5,
        "rB": 0.8,
        "KAA_Tref": 2.27,
        "dH_R": 1893.0,
    }
    # Three temperatures spanning ~Tsarevsky calibration range. Each
    # condition gets ~12 FA samples => 36 total.
    sim_conds = dict(
        A0=[1.0, 1.0, 1.0],
        B0=[1.0, 1.0, 1.0],
        T_K=[320.0, 350.0, 380.0],
    )
    t_evals = np.arange(0.05, 0.61, 0.05)

    model = BinaryReversible1TimeFit(observables=["FA"])
    # Drop priors on KAA_Tref / dH_R. The default model has Tsarevsky-NLS
    # priors that make pyPESTO wrap the AmiciObjective in a prior-aware
    # AggregatedObjective; simulate_problem then fails its
    # `isinstance(objective, AmiciObjective)` assertion. With 3-temperature
    # data the parameters are identifiable from the likelihood alone.
    model.fit_params["KAA_Tref"].prior_type = None
    model.fit_params["KAA_Tref"].prior_params = None
    model.fit_params["dH_R"].prior_type = None
    model.fit_params["dH_R"].prior_params = None

    fit = run_recovery(
        tmp_path=tmp_path,
        model=model,
        true_params=true_params,
        sim_conds=sim_conds,
        t_evals=t_evals,
        noise=0.01,
    )
    print(f"\nTrue: {true_params}")
    # log_tol is generous (0.10 dex ~ 25%): with priors on KAA_Tref and dH_R,
    # the posterior should pull these toward the truth, but with only 36 data
    # points the precision is limited. Adjust upward if seed-flaky.
    assert_recovered(fit, true_params, log_tol=0.10)
