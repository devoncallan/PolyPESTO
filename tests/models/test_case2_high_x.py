"""
High-conversion validation: Case II vs reversible_rxn up to the physical
equilibrium conversion x_eq.

reversible_rxn integrates real concentrations vs real time, so the system
naturally settles to its thermodynamic equilibrium x_eq < 1. Case II's
conversion-clock formulation doesn't know about x_eq -- the clock keeps
ticking past it -- so we only compare on x in [0.05, x_eq * 0.97] where
both models are answering the same chemistry question.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest
import amici

from .conftest import simulate as simulate_state


# Target x_total grid will be capped per-case at x_eq * 0.97.
X_TARGET_MIN = 0.05
N_TARGETS = 30

# Van't Hoff calibration (matches BinaryReversible1TimeFit defaults).
KAA_TREF = 2.27
DH_R = 1893.0
TREF = 350.0


def _kaa_of_T(T_K: float) -> float:
    return KAA_TREF * np.exp(-DH_R * (1.0 / T_K - 1.0 / TREF))


def _rxn_xA_trajectory(
    amici_model,
    *,
    rA: float, rB: float, A0: float, B0: float,
    KAA: float, KBA: float, kpAA: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run reversible_rxn long enough to reach equilibrium. Return monotonic
    (x_total, xA) arrays."""
    for name, val in [
        ("kpAA", kpAA), ("rA", rA), ("rB", rB), ("rX", 1.0),
        ("KAA", KAA), ("KAB", 0.0), ("KBA", KBA), ("KBB", 0.0),
    ]:
        amici_model.setParameterByName(name, float(val))
    amici_model.setFixedParameterByName("A0", float(A0))
    amici_model.setFixedParameterByName("B0", float(B0))

    t_eval = np.concatenate([np.linspace(0.0, 100.0, 200), np.logspace(2.0, 6.0, 400)])
    amici_model.setTimepoints(t_eval)
    solver = amici_model.getSolver()
    solver.setAbsoluteTolerance(1e-10)
    solver.setRelativeTolerance(1e-8)
    solver.setMaxSteps(500_000)

    rdata = amici.runAmiciSimulation(amici_model, solver)
    assert rdata.status == 0, f"reversible_rxn AMICI failed: {rdata.status}"

    state_ids = list(amici_model.getStateIds())
    x_arr = np.asarray(rdata.x)
    A = x_arr[:, state_ids.index("A")]
    B = x_arr[:, state_ids.index("B")]
    x_total = np.maximum.accumulate(1.0 - (A + B) / (A0 + B0))
    xA = 1.0 - A / A0
    return x_total, xA


CASES = [
    # Mild reversibility, symmetric -- should be the easiest high-x test.
    dict(name="sym_T350_fBA0.3", rA=1.0, rB=1.0, T_K=350.0, f_BA=0.3, A0=1.0, B0=1.0),
    # Stronger reversibility at higher T.
    dict(name="sym_T370_fBA0.5", rA=1.0, rB=1.0, T_K=370.0, f_BA=0.5, A0=1.0, B0=1.0),
    # Max f_BA at moderate T.
    dict(name="sym_T350_fBA1.0", rA=1.0, rB=1.0, T_K=350.0, f_BA=1.0, A0=1.0, B0=1.0),
    # Asymmetric rA, rB but moderate kd.
    dict(name="rA2_rB0.5_T350_fBA0.5", rA=2.0, rB=0.5, T_K=350.0, f_BA=0.5, A0=1.0, B0=1.0),
    # Asymmetric feed -- the case rev_ode broke on earliest.
    dict(name="A0_0.3_B0_0.7_fBA0.5", rA=1.0, rB=1.0, T_K=350.0, f_BA=0.5, A0=0.3, B0=0.7),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_case2_matches_reversible_rxn_to_xeq(amici_case2, amici_reversible_rxn, case):
    rA, rB = case["rA"], case["rB"]
    T_K, f_BA = case["T_K"], case["f_BA"]
    A0, B0 = case["A0"], case["B0"]

    KAA = _kaa_of_T(T_K)
    KBA = f_BA * KAA

    # Ground truth: reversible_rxn run to equilibrium.
    x_total_rxn, xA_rxn_t = _rxn_xA_trajectory(
        amici_reversible_rxn,
        rA=rA, rB=rB, A0=A0, B0=B0, KAA=KAA, KBA=KBA,
    )
    x_eq = float(x_total_rxn[-1])

    # Compare on [X_TARGET_MIN, 0.97 * x_eq]: stay clear of the equilibrium
    # plateau where Case II's clock-based formulation has no physical anchor.
    x_max = 0.97 * x_eq
    x_targets = np.linspace(X_TARGET_MIN, x_max, N_TARGETS)
    xA_rxn = np.interp(x_targets, x_total_rxn, xA_rxn_t)

    # Case II on the same grid.
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

    # Lowry approximation error grows with conversion; 1e-2 absolute (~1% on
    # xA) covers the regime up to 0.97 * x_eq.
    assert max_diff < 1e-2, (
        f"Case II vs reversible_rxn disagrees by {max_diff:.3e} at x={max_diff_x:.3f} "
        f"(x_eq={x_eq:.3f}) for "
        f"(rA={rA}, rB={rB}, T_K={T_K}, f_BA={f_BA}, A0={A0}, B0={B0})."
    )
