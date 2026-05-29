"""
Sanity test: irr_cpe (Mayo-Lewis closed form) should agree with irr_ode
(full kinetic ODE with explicit PA/PB chain-end species) at the same
(rA, rB, A0, B0).

If this test passes, we have two trustworthy irreversible references to
build the reversible Case II validation on top of. If it fails, the cpe
rescaling formulation in this codebase has its own numerical issues we
need to characterise before validating any reversible variant against it.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import simulate


# Compare in [1e-3, 0.6]: away from t=0 to avoid the +eps singularity in
# the conversion-clock formulation, and away from full conversion where
# 1/(dx_dt+eps) blows up.
X_GRID = np.linspace(1e-3, 0.6, 40)

CASES = [
    dict(name="symmetric_rA1_rB1", rA=1.0, rB=1.0, A0=1.0, B0=1.0),
    dict(name="rA_high",            rA=5.0, rB=0.5, A0=1.0, B0=1.0),
    dict(name="rB_high",            rA=0.5, rB=5.0, A0=1.0, B0=1.0),
    dict(name="asymmetric_feed",    rA=2.0, rB=0.5, A0=0.7, B0=0.3),
    dict(name="rich_in_B",          rA=1.0, rB=1.0, A0=0.3, B0=0.7),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_irr_cpe_matches_irr_ode(amici_irr_cpe, amici_irr_ode, case):
    """xA(x_total) trajectories should match between the two irreversible
    formulations."""
    common = dict(
        rA=case["rA"], rB=case["rB"], A0=case["A0"], B0=case["B0"],
        kpAA=1.0,
    )
    # irr_cpe parameters: rA, rB, plus dummy rX (unused in dA/dB of irr_cpe).
    cpe_params = dict(common, rX=1.0)
    # irr_ode uses kpAA/kpAB/kpBA/kpBB derived from define_irreversible_k
    # which sets kpAB = kpAA/rA, kpBB = kpAA/rX, kpBA = kpBB/rB.
    ode_params = dict(common, rX=1.0)

    out_cpe = simulate(amici_irr_cpe, X_GRID, cpe_params)
    out_ode = simulate(amici_irr_ode, X_GRID, ode_params)

    xA_cpe = out_cpe["xA"]
    xA_ode = out_ode["xA"]

    max_diff = float(np.max(np.abs(xA_cpe - xA_ode)))
    # Report intermediate values for diagnostics on failure.
    print(
        f"\n[{case['name']}] xA at x=0.3: cpe={np.interp(0.3, X_GRID, xA_cpe):.5f} "
        f"ode={np.interp(0.3, X_GRID, xA_ode):.5f}"
    )
    print(
        f"[{case['name']}] xA at x=0.5: cpe={np.interp(0.5, X_GRID, xA_cpe):.5f} "
        f"ode={np.interp(0.5, X_GRID, xA_ode):.5f}"
    )
    print(f"[{case['name']}] max |diff| = {max_diff:.3e}")

    # 1e-3 absolute = 0.1% of xA range; well below experimental noise.
    assert max_diff < 1e-3, (
        f"irr_cpe and irr_ode disagree by {max_diff:.3e} at some x in {X_GRID[0]}..{X_GRID[-1]}"
    )
