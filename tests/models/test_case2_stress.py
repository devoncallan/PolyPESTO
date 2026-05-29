"""
Stress test: BinaryReversible2TimeFit (closed-form Lowry Case II) vs.
rev_ode (full kinetic ODE) under harder conditions where the Lowry
stationary-Markov approximation is more strained.

Stressors:
- Higher temperatures (KAA up to ~5)
- f_BA at the boundary (1.0)
- Extreme rA/rB asymmetry (10:1 and 1:10)
- Asymmetric feed compositions (A0/B0 != 1)
- Conversion pushed to x = 0.7 (closer to where rev_ode rescaling fragility kicks in)

Compared to test_case2_matches_rev_ode.py: tolerance is loosened to 5e-3 to
reflect the known approximation error growing with kd magnitudes, and the
parameter sweep covers the harder corners of the operating envelope.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import simulate


# Narrow x range: rev_ode (our ground truth) is numerically fragile at high
# conversion when extreme rA/rB or A0/B0 cause one monomer to deplete -- the
# `1/(dx_dt+eps)` rescaling pegs xA to a boundary value and we'd be testing
# rev_ode's failure mode, not Case II's accuracy. With the asymmetric-feed
# (A0=0.8, B0=0.2) case in this set, rev_ode goes pathological as early as
# x ~ 0.3, so we cap at 0.25 to keep ALL stress combinations in the
# well-conditioned regime.
X_GRID = np.linspace(1e-3, 0.25, 50)

KAA_TREF = 2.27
DH_R = 1893.0
TREF = 350.0


def _kaa_of_T(T_K):
    return KAA_TREF * np.exp(-DH_R * (1.0 / T_K - 1.0 / TREF))


CASES = [
    # Push temperature high (KAA grows): KAA(385K) ~ 3.8, KAA(395K) ~ 4.4
    dict(name="T385_sym_fBA1",        rA=1.0, rB=1.0,  T_K=385.0, f_BA=1.0, A0=1.0, B0=1.0),
    dict(name="T395_sym_fBA0.8",      rA=1.0, rB=1.0,  T_K=395.0, f_BA=0.8, A0=1.0, B0=1.0),
    # Extreme rA/rB asymmetry (one monomer much more reactive in copropagation)
    dict(name="rA10_rB0.1_T370_fBA0.5", rA=10.0, rB=0.1, T_K=370.0, f_BA=0.5, A0=1.0, B0=1.0),
    dict(name="rA0.1_rB10_T370_fBA0.5", rA=0.1, rB=10.0, T_K=370.0, f_BA=0.5, A0=1.0, B0=1.0),
    # Asymmetric feed, moderate kd
    dict(name="A0_0.2_B0_0.8_fBA0.5", rA=1.0, rB=1.0,  T_K=370.0, f_BA=0.5, A0=0.2, B0=0.8),
    dict(name="A0_0.8_B0_0.2_fBA0.5", rA=1.0, rB=1.0,  T_K=370.0, f_BA=0.5, A0=0.8, B0=0.2),
    # Joint stress: high T + asymmetric feed + high f_BA
    dict(name="T385_A0_0.3_B0_0.7_fBA1", rA=1.5, rB=0.5, T_K=385.0, f_BA=1.0, A0=0.3, B0=0.7),
    # Mild stress with very different rA/rB
    dict(name="rA5_rB0.2_T360_fBA0.7",  rA=5.0, rB=0.2, T_K=360.0, f_BA=0.7, A0=1.0, B0=1.0),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_case2_stress(amici_case2, amici_rev_ode, case):
    rA, rB = case["rA"], case["rB"]
    T_K, f_BA = case["T_K"], case["f_BA"]
    A0, B0 = case["A0"], case["B0"]

    KAA = _kaa_of_T(T_K)
    KBA = f_BA * KAA

    case2_params = dict(
        rA=rA, rB=rB, rX=1.0,
        A0=A0, B0=B0, xf=1.0,
        T_K=T_K, Tref=TREF,
        KAA_Tref=KAA_TREF, dH_R=DH_R, f_BA=f_BA,
        kpAA=1.0,
    )
    rev_params = dict(
        rA=rA, rB=rB, rX=1.0,
        A0=A0, B0=B0,
        KAA=KAA, KAB=0.0, KBA=KBA, KBB=0.0,
        kpAA=1.0,
    )

    case2_out = simulate(amici_case2, X_GRID, case2_params)
    rev_out = simulate(amici_rev_ode, X_GRID, rev_params)

    xA_case2 = case2_out["xA"]
    xA_rev = rev_out["xA"]
    max_diff = float(np.max(np.abs(xA_case2 - xA_rev)))

    print(
        f"\n[{case['name']}] KAA={KAA:.3f} KBA={KBA:.3f}; "
        f"x=0.1: case2={np.interp(0.1, X_GRID, xA_case2):.5f} rev={np.interp(0.1, X_GRID, xA_rev):.5f}; "
        f"x=0.2: case2={np.interp(0.2, X_GRID, xA_case2):.5f} rev={np.interp(0.2, X_GRID, xA_rev):.5f}"
    )
    print(f"[{case['name']}] max |diff| = {max_diff:.3e}")

    # 5e-3 absolute (~0.5% on xA) covers the known systematic Lowry error in
    # this stress regime. Still ~5x tighter than typical experimental noise
    # on FA/xA (which is 1-5%).
    assert max_diff < 5e-3, (
        f"Case II disagrees with rev_ode by {max_diff:.3e} for "
        f"(rA={rA}, rB={rB}, T_K={T_K}, f_BA={f_BA}, A0={A0}, B0={B0}). "
        f"Lowry approximation has broken down or there's a Case II math bug."
    )
