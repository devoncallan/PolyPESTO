"""
Shared test infrastructure for model-level numerical validation.

Compiles SBML models to AMICI in a session-scoped cache directory so the
first test that needs a model pays the build cost; subsequent tests reuse
the compiled .so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pytest

import amici

# Touch polypesto.core to satisfy the package init order before any
# polypesto.models.* import runs.
import polypesto.core  # noqa: F401


# Persistent build cache: reuse the repo's amici_models/<amici_version>/
# directory so test builds live alongside the regular model artifacts.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD_CACHE = _REPO_ROOT / "amici_models" / amici.__version__


def _compile(sbml_model, model_name: str, constants: Iterable[str]):
    """Compile (or reuse cached build) and return an AMICI Model instance.

    Layout matches the rest of the repo: amici_models/<amici_ver>/<model_name>/
    """
    out_dir = _BUILD_CACHE / model_name
    cached_so = list((out_dir / model_name).glob("_*.so")) if out_dir.exists() else []

    if not cached_so:
        _BUILD_CACHE.mkdir(parents=True, exist_ok=True)
        sbml_path = _BUILD_CACHE / f"{model_name}.xml"
        sbml_model.to_file(str(sbml_path))
        importer = amici.SbmlImporter(str(sbml_path))
        # simplify=None: skip the sympy simplification pass that blows up on
        # piecewise(...) rules. Forward simulation only; we don't need
        # sensitivities here.
        importer.sbml2amici(
            model_name=model_name,
            output_dir=str(out_dir),
            constant_parameters=list(constants),
            verbose=False,
            simplify=None,
        )

    module = amici.import_model_module(model_name, str(out_dir))
    return module.getModel()


def _set_if_present(model, name: str, value: float) -> bool:
    if name in list(model.getParameterIds()):
        model.setParameterByName(name, float(value))
        return True
    if name in list(model.getFixedParameterIds()):
        model.setFixedParameterByName(name, float(value))
        return True
    return False


def simulate(
    model, x_grid: np.ndarray, params: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """Run a forward simulation; return state trajectories keyed by state id."""
    for k, v in params.items():
        _set_if_present(model, k, v)

    model.setTimepoints(x_grid)
    solver = model.getSolver()
    solver.setAbsoluteTolerance(1e-10)
    solver.setRelativeTolerance(1e-8)
    solver.setMaxSteps(100_000)

    rdata = amici.runAmiciSimulation(model, solver)
    assert rdata.status == 0, f"AMICI failed: status {rdata.status}"

    state_ids = list(model.getStateIds())
    return {sid: np.array(rdata.x[:, i]) for i, sid in enumerate(state_ids)}


# ---- Model fixtures (session-scoped: compile once, reuse across all tests) ----


@pytest.fixture(scope="session")
def amici_irr_cpe():
    from polypesto.models.binary.irreversible import irr_cpe
    return _compile(irr_cpe(), "irr_cpe", constants=["A0", "B0"])


@pytest.fixture(scope="session")
def amici_irr_ode():
    from polypesto.models.binary.irreversible import irr_ode
    return _compile(irr_ode(), "irr_ode", constants=["A0", "B0"])


@pytest.fixture(scope="session")
def amici_case1():
    from polypesto.models.binary.reversible1_time_fit import (
        rev_cpe_lowry_caseI_time_fit,
    )
    return _compile(
        rev_cpe_lowry_caseI_time_fit(),
        "rev_cpe_lowry_caseI_time_fit",
        constants=["T_K", "A0", "B0", "xf", "Tref"],
    )


@pytest.fixture(scope="session")
def amici_case2():
    from polypesto.models.binary.reversible2_time_fit import (
        rev_cpe_lowry_caseII_time_fit,
    )
    return _compile(
        rev_cpe_lowry_caseII_time_fit(),
        "rev_cpe_lowry_caseII_time_fit",
        constants=["T_K", "A0", "B0", "xf", "Tref"],
    )


@pytest.fixture(scope="session")
def amici_rev_ode():
    from polypesto.models.binary.reversible import rev_ode
    return _compile(rev_ode(), "rev_ode", constants=["A0", "B0"])


@pytest.fixture(scope="session")
def amici_reversible_rxn():
    """Reaction-network SBML model. Integrates real concentrations against
    real time -- no conversion-clock rescaling, so it stays well-conditioned
    at high conversion where rev_ode pegs to monomer-depletion boundaries.

    NOT a valid PEtab model (real time != measurement time), so we use it
    only as a forward simulator inside tests."""
    from polypesto.models.binary.reversible import reversible_rxn
    return _compile(reversible_rxn(), "reversible_rxn", constants=["A0", "B0"])


# ---- Parameter-estimation recovery helper --------------------------------


def run_recovery(
    *,
    tmp_path: Path,
    model,
    true_params: Mapping[str, float],
    sim_conds: Mapping[str, List[float]],
    t_evals,
    noise: float,
    n_starts: int = 50,
    n_samples: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Simulate noisy data from `true_params`, fit, return best-fit param dict
    (unscaled physical values).

    NOTE: do NOT pass obs_noise on the model -- doing so triggers PEtab lint
    failure due to a conflict between the literal noiseFormula on the
    observable and the per-measurement noiseParameters override (see
    test_recovery_irreversible.py for the gory detail). The model must be
    instantiated with obs_noise=None and noise applied through `meas_noise`
    on create_sim_conditions, which is what this helper enforces.
    """
    from petab.v1.parameters import unscale

    from polypesto.core import create_sim_conditions, simulate_problem

    # Lock noise realization so the test is reproducible.
    np.random.seed(seed)

    conds = create_sim_conditions(
        true_params=dict(true_params),
        conds=dict(sim_conds),
        t_evals=t_evals,
        meas_noise=noise,
    )
    problem = simulate_problem(
        prob_dir=tmp_path / "recovery_problem",
        model=model,
        conds=conds,
        overwrite=True,
    )
    result = problem.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=n_starts, method="Nelder-Mead"),
            sample=dict(n_samples=n_samples, n_chains=3),
        ),
        overwrite=True,
        save=False,
        plot=False,
    )
    assert result is not None, "parameter estimation returned None"

    # result.optimize_result.x[0] is the best start's parameter vector in
    # pyPESTO's internal scaled coordinates. Unscale to physical values.
    pp = result.problem
    best_x = result.optimize_result.x[0]
    return {
        name: float(unscale(best_x[i], pp.x_scales[i]))
        for i, name in enumerate(pp.x_names)
    }


def assert_recovered(
    fit: Mapping[str, float],
    true_params: Mapping[str, float],
    *,
    log_tol: float = 0.05,
    lin_tol: Optional[Dict[str, float]] = None,
) -> None:
    """Assert each true_params entry is recovered.

    By default uses |log10(fit/true)| < log_tol. For parameters listed in
    `lin_tol` (mapping name -> absolute tol), use absolute tolerance instead
    -- needed for params that can legitimately be 0 (e.g. f_BA with
    estimate=False).
    """
    lin_tol = lin_tol or {}
    for name, true_val in true_params.items():
        fit_val = fit[name]
        if name in lin_tol:
            err = abs(fit_val - true_val)
            tol = lin_tol[name]
            print(f"  {name}: true={true_val:.4f} fit={fit_val:.4f} |abs|={err:.4f}")
            assert err < tol, (
                f"Parameter {name} not recovered: true={true_val} fit={fit_val} "
                f"|abs|={err:.4f} (threshold {tol})"
            )
        else:
            log_err = abs(np.log10(fit_val / true_val))
            print(f"  {name}: true={true_val:.4f} fit={fit_val:.4f} "
                  f"|log10(fit/true)|={log_err:.4f}")
            assert log_err < log_tol, (
                f"Parameter {name} not recovered: true={true_val} fit={fit_val} "
                f"|log10 ratio|={log_err:.4f} (threshold {log_tol})"
            )
