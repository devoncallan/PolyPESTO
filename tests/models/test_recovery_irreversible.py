"""
Parameter-estimation recovery test on BinaryIrreversible (2 free params).
Calibration baseline for the recovery infrastructure.

Marked `slow`. Run explicitly with:
    pytest -m slow tests/models/test_recovery_irreversible.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polypesto.models.binary import BinaryIrreversible

from .conftest import assert_recovered, run_recovery


pytestmark = pytest.mark.slow


def test_irreversible_recovery(tmp_path: Path):
    true_params = {"rA": 2.0, "rB": 0.5}
    sim_conds = dict(A0=[0.70, 0.30], B0=[0.30, 0.70])
    t_evals = np.arange(0.05, 0.61, 0.05)  # 12 points per condition

    model = BinaryIrreversible(observables=["FA"])

    fit = run_recovery(
        tmp_path=tmp_path,
        model=model,
        true_params=true_params,
        sim_conds=sim_conds,
        t_evals=t_evals,
        noise=0.01,
    )
    print(f"\nTrue: {true_params}")
    assert_recovered(fit, true_params, log_tol=0.05)
