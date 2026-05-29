"""
High-conversion stress test: Case II vs reversible_rxn under
extreme rA/rB asymmetry, strong reversibility, and asymmetric feeds.

Pushes the Lowry stationary-Markov approximation harder than
test_case2_high_x.py while still using reversible_rxn (no rev_ode
high-x pathology). Comparison runs up to 0.97 * x_eq per case.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest
import amici

from .conftest import simulate as simulate_state
from .test_case2_high_x import _rxn_xA_trajectory, _kaa_of_T, TREF, KAA_TREF, DH_R


X_TARGET_MIN = 0.05
N_TARGETS = 30


CASES = [
    # Extreme rA/rB asymmetry
    dict(name="rA10_rB0.1_T350_fBA0.5", rA=10.0, rB=0.1, T_K=350.0, f_BA=0.5, A0=1.0, B0=1.0),
    dict(name="rA0.1_rB10_T350_fBA0.5", rA=0.1, rB=10.0, T_K=350.0, f_BA=0.5, A0=1.0, B0=1.0),
    dict(name="rA20_rB0.05_T350_fBA0.5", rA=20.0, rB=0.05, T_K=350.0, f_BA=0.5, A0=1.0, B0=1.0),

    # Strong reversibility (high T => large KAA; max f_BA)
    dict(name="sym_T395_fBA1.0",  rA=1.0, rB=1.0, T_K=395.0, f_BA=1.0, A0=1.0, B0=1.0),
    dict(name="sym_T400_fBA1.0",  rA=1.0, rB=1.0, T_K=400.0, f_BA=1.0, A0=1.0, B0=1.0),

    # Asymmetric feeds (rich/lean in each monomer)
    dict(name="A0_0.1_B0_0.9_T350_fBA0.5", rA=1.0, rB=1.0, T_K=350.0, f_BA=0.5, A0=0.1, B0=0.9),
    dict(name="A0_0.9_B0_0.1_T350_fBA0.5", rA=1.0, rB=1.0, T_K=350.0, f_BA=0.5, A0=0.9, B0=0.1),

    # Joint: extreme rA/rB + asymmetric feed + strong reversibility
    dict(name="rA5_rB0.2_A0_0.2_B0_0.8_T380_fBA0.8",
         rA=5.0, rB=0.2, T_K=380.0, f_BA=0.8, A0=0.2, B0=0.8),
    dict(name="rA0.2_rB5_A0_0.8_B0_0.2_T380_fBA0.8",
         rA=0.2, rB=5.0, T_K=380.0, f_BA=0.8, A0=0.8, B0=0.2),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_case2_vs_reversible_rxn_stress(amici_case2, amici_reversible_rxn, case):
    rA, rB = case["rA"], case["rB"]
    T_K, f_BA = case["T_K"], case["f_BA"]
    A0, B0 = case["A0"], case["B0"]

    KAA = _kaa_of_T(T_K)
    KBA = f_BA * KAA

    x_total_rxn, xA_rxn_t = _rxn_xA_trajectory(
        amici_reversible_rxn,
        rA=rA, rB=rB, A0=A0, B0=B0, KAA=KAA, KBA=KBA,
    )
    x_eq = float(x_total_rxn[-1])
    x_max = 0.97 * x_eq
    x_targets = np.linspace(X_TARGET_MIN, x_max, N_TARGETS)
    xA_rxn = np.interp(x_targets, x_total_rxn, xA_rxn_t)

    case2_params: Dict[str, float] = dict(
        rA=rA, rB=rB, rX=1.0, A0=A0, B0=B0, xf=1.0,
        T_K=T_K, Tref=TREF,
        KAA_Tref=KAA_TREF, dH_R=DH_R, f_BA=f_BA,
        kpAA=1.0,
    )
    case2_out = simulate_state(amici_case2, x_targets, case2_params)
    xA_case2 = case2_out["xA"]

    diffs = np.abs(xA_case2 - xA_rxn)
    max_diff = float(np.max(diffs))
    max_diff_x = float(x_targets[int(np.argmax(diffs))])

    print(f"\n[{case['name']}] KAA={KAA:.3f} KBA={KBA:.3f} x_eq={x_eq:.3f} comparing to x={x_max:.3f}")
    for x_check in (0.1, 0.3, 0.5, x_max * 0.99):
        if x_check > x_max:
            continue
        i = int(np.argmin(np.abs(x_targets - x_check)))
        print(
            f"  x={x_targets[i]:.3f}: case2={xA_case2[i]:.5f} rxn={xA_rxn[i]:.5f} "
            f"diff={xA_case2[i] - xA_rxn[i]:+.3e}"
        )
    print(f"  max |diff| = {max_diff:.3e} at x={max_diff_x:.3f}")

    # Stress regime tolerance: 2e-2 absolute (~2% on xA), still well below
    # typical experimental noise on FA/xA.
    assert max_diff < 2e-2, (
        f"Case II vs reversible_rxn disagrees by {max_diff:.3e} at x={max_diff_x:.3f} "
        f"(x_eq={x_eq:.3f}) for "
        f"(rA={rA}, rB={rB}, T_K={T_K}, f_BA={f_BA}, A0={A0}, B0={B0})."
    )
