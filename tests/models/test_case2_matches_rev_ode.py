"""
Validate the BinaryReversible2TimeFit closed-form (Lowry Case II) against
rev_ode (the full kinetic ODE with explicit PAA/PAB/PBA/PBB dyad species)
in the Case II regime: kdAA != 0, kdBA != 0, kdAB = kdBB = 0.

This is the test that actually exercises my Case II math (the modified pA
and dA expressions). Case I in irreversible limit and Case II in
irreversible limit are already validated in test_irreversible_limit.py.

rev_ode parameter convention:
    KAA, KBA driven via Van't Hoff: KAA(T) = KAA_Tref * exp(-dH_R*(1/T-1/Tref))
                                    KBA    = f_BA * KAA
    KAB, KBB = 0 (B side fully irreversible, Case II definition)
    kdXY = kpXY * KXY (from define_reversible_k)
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import simulate


# Compare in [1e-3, 0.6]: 1/(dx_dt+eps) in rev_ode breaks down near
# full conversion. The Lowry closed-form is fine across the whole range
# but we need both sides to be well-conditioned to compare them.
X_GRID = np.linspace(1e-3, 0.6, 50)

# Van't Hoff calibration matching the BinaryReversible1TimeFit defaults.
KAA_TREF = 2.27
DH_R = 1893.0
TREF = 350.0


def _kaa_of_T(T_K):
    return KAA_TREF * np.exp(-DH_R * (1.0 / T_K - 1.0 / TREF))


# Cover (rA, rB) asymmetry, temperature (KAA(T) range), and f_BA span.
CASES = [
    dict(name="sym_T350_fBA0.3", rA=1.0, rB=1.0, T_K=350.0, f_BA=0.3),
    dict(name="sym_T350_fBA0.5", rA=1.0, rB=1.0, T_K=350.0, f_BA=0.5),
    dict(name="sym_T350_fBA1.0", rA=1.0, rB=1.0, T_K=350.0, f_BA=1.0),
    dict(name="rA2_rB0.5_T370",  rA=2.0, rB=0.5, T_K=370.0, f_BA=0.5),
    dict(name="rA0.5_rB2_T330",  rA=0.5, rB=2.0, T_K=330.0, f_BA=0.5),
    dict(name="rich_in_B_fBA0.8",rA=1.0, rB=1.0, T_K=355.0, f_BA=0.8),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_case2_matches_rev_ode(amici_case2, amici_rev_ode, case):
    rA, rB, T_K, f_BA = case["rA"], case["rB"], case["T_K"], case["f_BA"]
    KAA = _kaa_of_T(T_K)
    KBA = f_BA * KAA

    case2_params = dict(
        rA=rA, rB=rB, rX=1.0,
        A0=1.0, B0=1.0, xf=1.0,
        T_K=T_K, Tref=TREF,
        KAA_Tref=KAA_TREF, dH_R=DH_R, f_BA=f_BA,
        kpAA=1.0,
    )
    rev_params = dict(
        rA=rA, rB=rB, rX=1.0,
        A0=1.0, B0=1.0,
        KAA=KAA, KAB=0.0, KBA=KBA, KBB=0.0,
        kpAA=1.0,
    )

    case2_out = simulate(amici_case2, X_GRID, case2_params)
    rev_out = simulate(amici_rev_ode, X_GRID, rev_params)

    xA_case2 = case2_out["xA"]
    xA_rev = rev_out["xA"]
    max_diff = float(np.max(np.abs(xA_case2 - xA_rev)))

    print(
        f"\n[{case['name']}] KAA={KAA:.3f} KBA={KBA:.3f} "
        f"x=0.3: case2={np.interp(0.3, X_GRID, xA_case2):.5f} "
        f"rev={np.interp(0.3, X_GRID, xA_rev):.5f}; "
        f"x=0.5: case2={np.interp(0.5, X_GRID, xA_case2):.5f} "
        f"rev={np.interp(0.5, X_GRID, xA_rev):.5f}"
    )
    print(f"[{case['name']}] max |diff| = {max_diff:.3e}")

    # The Lowry stationary-Markov approximation is asymptotic in chain length.
    # Empirically, in the regime spanned by these cases (T up to 370 K, KAA up
    # to ~3, f_BA up to 1), the closed form matches rev_ode to ~1e-3 with the
    # disagreement biased systematically (case2 slightly under-predicts xA),
    # consistent with the known O(1/Nchain) error of the stationary
    # assumption. 2e-3 covers the observed worst case (rA=2, rB=0.5, T=370)
    # with margin. This is two orders of magnitude tighter than typical
    # experimental measurement noise on FA / xA.
    assert max_diff < 2e-3, (
        f"Case II disagrees with rev_ode by {max_diff:.3e} for "
        f"(rA={rA}, rB={rB}, T_K={T_K}, f_BA={f_BA}). "
        f"Either Case II math is wrong or Lowry approximation has broken down."
    )
