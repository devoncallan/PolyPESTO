"""
Every reversible model must reduce to irr_cpe (the canonical Mayo-Lewis
closed form) when its depropagation constants go to zero.

This is the foundational sanity check before any reversibility-specific
validation: if a reversible model can't match the irreversible limit, the
model is wrong and no further test result on it is trustworthy.

Models under test:
    - BinaryReversible1TimeFit  (Case I)            -> set KAA_Tref = 0
    - BinaryReversible2TimeFit  (Case II)           -> set KAA_Tref = 0 (kills both kdAA and kdBA via KAA chain)
    - BinaryReversible          (rev_ode, full)     -> set KAA = KAB = KBA = KBB = 0
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import simulate


# Stay well clear of full conversion: 1/(dx_dt+eps) in the rate rules is
# fragile near total monomer depletion.
X_GRID = np.linspace(1e-3, 0.6, 50)

CASES = [
    dict(name="symmetric_rA1_rB1", rA=1.0, rB=1.0, A0=1.0, B0=1.0),
    dict(name="rA_high",            rA=5.0, rB=0.5, A0=1.0, B0=1.0),
    dict(name="rB_high",            rA=0.5, rB=5.0, A0=1.0, B0=1.0),
    dict(name="asymmetric_feed",    rA=2.0, rB=0.5, A0=0.7, B0=0.3),
    dict(name="rich_in_B",          rA=1.0, rB=1.0, A0=0.3, B0=0.7),
]


def _irr_params(case):
    return dict(
        rA=case["rA"], rB=case["rB"], rX=1.0,
        A0=case["A0"], B0=case["B0"],
        kpAA=1.0,
    )


def _diff(xa, xb):
    return float(np.max(np.abs(xa - xb)))


# ---- Case I in irreversible limit ----

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_case1_matches_irr_cpe_when_KAA_zero(amici_irr_cpe, amici_case1, case):
    """KAA_Tref = 0 forces KAA = 0 (Van't Hoff KAA_Tref * exp(...) = 0)
    which forces kdAA = kpAA*KAA = 0. Case I must reduce to irr_cpe."""
    irr_out = simulate(amici_irr_cpe, X_GRID, _irr_params(case))
    case1_params = dict(
        rA=case["rA"], rB=case["rB"], rX=1.0,
        A0=case["A0"], B0=case["B0"], xf=1.0,
        T_K=350.0, Tref=350.0,
        KAA_Tref=0.0,    # <- forces KAA(T) = 0 => kdAA = 0
        dH_R=1893.0,     # arbitrary; multiplied by 0
        kpAA=1.0,
    )
    case1_out = simulate(amici_case1, X_GRID, case1_params)

    d = _diff(irr_out["xA"], case1_out["xA"])
    print(f"\n[{case['name']}] case1 vs irr_cpe: max |diff| = {d:.3e}")
    assert d < 1e-3, f"Case I (KAA=0) doesn't reduce to irr_cpe: max |diff| = {d:.3e}"


# ---- Case II in irreversible limit ----

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("f_BA", [0.0, 0.5, 1.0], ids=["fBA=0", "fBA=0.5", "fBA=1"])
def test_case2_matches_irr_cpe_when_KAA_zero(amici_irr_cpe, amici_case2, case, f_BA):
    """KAA_Tref = 0 => KAA = 0 => KBA = f_BA*KAA = 0 => kdAA = kdBA = 0.
    Case II must reduce to irr_cpe regardless of f_BA."""
    irr_out = simulate(amici_irr_cpe, X_GRID, _irr_params(case))
    case2_params = dict(
        rA=case["rA"], rB=case["rB"], rX=1.0,
        A0=case["A0"], B0=case["B0"], xf=1.0,
        T_K=350.0, Tref=350.0,
        KAA_Tref=0.0,    # kills both kdAA and (via KBA = f_BA*KAA) kdBA
        dH_R=1893.0,
        f_BA=f_BA,       # immaterial when KAA = 0
        kpAA=1.0,
    )
    case2_out = simulate(amici_case2, X_GRID, case2_params)

    d = _diff(irr_out["xA"], case2_out["xA"])
    print(f"\n[{case['name']} f_BA={f_BA}] case2 vs irr_cpe: max |diff| = {d:.3e}")
    assert d < 1e-3, (
        f"Case II (KAA=0, f_BA={f_BA}) doesn't reduce to irr_cpe: max |diff| = {d:.3e}"
    )


# ---- rev_ode (full kinetic ODE) in irreversible limit ----

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_rev_ode_matches_irr_cpe_when_all_K_zero(amici_irr_cpe, amici_rev_ode, case):
    """rev_ode has KAA, KAB, KBA, KBB as direct parameters. Zeroing all of
    them makes kdAA = kdAB = kdBA = kdBB = 0 (fully irreversible)."""
    irr_out = simulate(amici_irr_cpe, X_GRID, _irr_params(case))
    rev_params = dict(
        rA=case["rA"], rB=case["rB"], rX=1.0,
        A0=case["A0"], B0=case["B0"],
        KAA=0.0, KAB=0.0, KBA=0.0, KBB=0.0,
        kpAA=1.0,
    )
    rev_out = simulate(amici_rev_ode, X_GRID, rev_params)

    d = _diff(irr_out["xA"], rev_out["xA"])
    print(f"\n[{case['name']}] rev_ode vs irr_cpe: max |diff| = {d:.3e}")
    assert d < 1e-3, (
        f"rev_ode (all K=0) doesn't reduce to irr_cpe: max |diff| = {d:.3e}"
    )
