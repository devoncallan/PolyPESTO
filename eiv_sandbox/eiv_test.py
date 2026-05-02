"""Standalone EIV (errors-in-variables) test for polypesto.

Toy ODE: dy/dt = k * c, y(0) = 0, integrated on t in [0, 1].
Analytic: y(t) = k * c * t.

Each "experiment" is one condition with its own true `c`. The c value is
"measured" with additive Gaussian noise; y is observed at several time points
with its own additive Gaussian noise.

We compare three fits for the global rate constant k:

  1. c_measured  -- c fixed in the condition table at its noisy measured value.
                    Standard regression. Should be biased when sigma_c > 0.
  2. c_oracle    -- c fixed at its true (unknown in practice) value. Best case.
  3. c_estimated -- c per-condition is itself a fit parameter with a normal
                    prior centred on the measured value (width = sigma_c).
                    This is the EIV setup.

Run:  python eiv_sandbox/eiv_test.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import petab.v1.C as PC

from polypesto.core import Problem
from polypesto.core import petab as pet
from polypesto.models import ModelBase, sbml

SANDBOX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SANDBOX_DIR / "_results"


# ---------------------------------------------------------------------------
# Toy model: dy/dt = k * c
# ---------------------------------------------------------------------------


class EivToyModel(ModelBase):
    def _default_obs(self) -> List[str]:
        return ["y"]

    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
        # Placeholder; real parameter table is written per-variant.
        return {
            "k": pet.FitParameter(
                id="k",
                scale=PC.LIN,
                bounds=(0.01, 10.0),
                nominal_value=1.0,
                estimate=True,
            ),
        }

    def _default_sbml_model(self) -> sbml.ModelDefinition:
        return _eiv_toy_sbml()

    def _default_solver_options(self, solver):
        solver.setRelativeTolerance(1e-9)
        solver.setAbsoluteTolerance(1e-11)
        solver.setSensitivityMethod(1)
        solver.setSensitivityOrder(1)
        return solver


def _eiv_toy_sbml() -> sbml.ModelDefinition:
    document, model = sbml.init_model("eiv_toy")
    sbml.create_compartment(model, "env", spatialDimensions=0, units="dimensionless")
    sbml.create_parameter(model, "k", value=1.0, constant=False)
    sbml.create_parameter(model, "c", value=1.0, constant=True)
    sbml.create_species(model, "y", initialAmount=0.0)
    sbml.create_rate_rule(model, "y", "k * c")
    return sbml.create_model(model, document)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def generate_data(
    k_true: float,
    c_true: np.ndarray,
    t_obs: np.ndarray,
    sigma_y: float,
    sigma_c: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = k_true * c_true[:, None] * t_obs[None, :]
    y_obs = y_true + rng.normal(0.0, sigma_y, size=y_true.shape)
    c_obs = c_true + rng.normal(0.0, sigma_c, size=c_true.shape)
    return y_obs, c_obs


# ---------------------------------------------------------------------------
# PEtab assembly per variant
# ---------------------------------------------------------------------------


def build_petab_dfs(
    variant: str,
    cond_ids: List[str],
    c_true: np.ndarray,
    c_obs: np.ndarray,
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    sigma_y: float,
    sigma_c_prior: float,
    k_init: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs_df = pet.utils.obs.define({"y": "y"})

    # Condition table: column "c" is either a numeric value or a parameter id
    cond_rows: List[Dict[str, object]] = []
    for i, cid in enumerate(cond_ids):
        if variant == "c_estimated":
            cval: object = f"c_{cid}"
        elif variant == "c_oracle":
            cval = float(c_true[i])
        elif variant == "c_measured":
            cval = float(c_obs[i])
        else:
            raise ValueError(variant)
        cond_rows.append(
            {PC.CONDITION_ID: cid, PC.CONDITION_NAME: cid, "c": cval}
        )
    cond_df = pet.utils.cond.format(pd.DataFrame(cond_rows))

    # Parameter table
    params: Dict[str, pet.FitParameter] = {
        "k": pet.FitParameter(
            id="k",
            scale=PC.LIN,
            bounds=(0.01, 10.0),
            nominal_value=k_init,
            estimate=True,
        ),
    }
    if variant == "c_estimated":
        for i, cid in enumerate(cond_ids):
            params[f"c_{cid}"] = pet.FitParameter(
                id=f"c_{cid}",
                scale=PC.LIN,
                bounds=(1e-3, 5.0),
                nominal_value=float(c_obs[i]),
                estimate=True,
                # PEtab v1 spec: objectivePriorType = "normal", parameters = "mean;std"
                prior_type="normal",
                prior_params=f"{float(c_obs[i])};{float(sigma_c_prior)}",
            )
    param_df = pet.utils.param.define(params)

    # Measurement table
    obs_id = "obs_y"  # polypesto's ID.obs_id("y")
    rows: List[Dict[str, object]] = []
    for i, cid in enumerate(cond_ids):
        for j, t in enumerate(t_obs):
            rows.append(
                {
                    PC.OBSERVABLE_ID: obs_id,
                    PC.SIMULATION_CONDITION_ID: cid,
                    PC.TIME: float(t),
                    PC.MEASUREMENT: float(y_obs[i, j]),
                    PC.NOISE_PARAMETERS: float(sigma_y),
                }
            )
    meas_df = pet.utils.meas.format(pd.DataFrame(rows))

    return obs_df, cond_df, param_df, meas_df


def write_problem_dir(prob_dir: Path, model: ModelBase, dfs) -> None:
    obs_df, cond_df, param_df, meas_df = dfs
    petab_data = pet.PetabData(obs_df, cond_df, param_df, meas_df)
    petab_data.write(prob_dir, model.sbml_model)


# ---------------------------------------------------------------------------
# Run a single fit and extract estimated k
# ---------------------------------------------------------------------------


def fit_variant(prob_dir: Path, model: ModelBase, n_starts: int = 10) -> Dict[str, float]:
    """Run optimization directly via pypesto with a single-process engine.

    Bypasses polypesto.run_parameter_estimation to avoid the MultiProcessEngine
    startup cost for this tiny problem and to suppress saving/plotting.
    """
    import pypesto.optimize as opt  # type: ignore
    import pypesto.engine as engine  # type: ignore

    prob = Problem.load(prob_dir, model)

    optimizer = opt.ScipyOptimizer(method="L-BFGS-B")
    result = opt.minimize(
        problem=prob.pypesto_problem,
        optimizer=optimizer,
        n_starts=n_starts,
        engine=engine.SingleCoreEngine(),
        progress_bar=False,
    )
    best = result.optimize_result.list[0]
    x_names = list(prob.pypesto_problem.x_names)
    x_dict = dict(zip(x_names, np.asarray(best.x).tolist()))

    # All params here are LIN-scaled, so x_dict values are raw values
    return {"fval": float(best.fval), "x": x_dict}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    seed = 7
    k_true = 1.5
    # 8 conditions with true c spread over [0.2, 1.0]
    c_true = np.linspace(0.2, 1.0, 8)
    t_obs = np.array([0.25, 0.5, 0.75, 1.0])
    sigma_y = 0.05  # absolute noise on y
    sigma_c = 0.15  # absolute noise on measured c (~25% of mean(c))

    cond_ids = [f"cond{i + 1}" for i in range(c_true.size)]
    y_obs, c_obs = generate_data(k_true, c_true, t_obs, sigma_y, sigma_c, seed=seed)

    print(f"k_true  = {k_true}")
    print(f"c_true  = {[round(v, 4) for v in c_true.tolist()]}")
    print(f"c_obs   = {[round(v, 4) for v in c_obs.tolist()]}")
    print(f"sigma_y = {sigma_y}, sigma_c = {sigma_c}")

    # Closed-form OLS sanity check (independent of polypesto):
    #   k_hat = sum(y * c * t) / sum((c*t)^2)
    def ols_k(c_vec: np.ndarray) -> float:
        ct = c_vec[:, None] * t_obs[None, :]
        return float((y_obs * ct).sum() / (ct ** 2).sum())

    print(f"OLS  k_hat (c=measured) = {ols_k(c_obs):.4f}   <-- expected attenuation bias")
    print(f"OLS  k_hat (c=true)     = {ols_k(c_true):.4f}   <-- oracle baseline")
    print()

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)

    summary_rows: List[Dict[str, object]] = []
    for variant in ["c_measured", "c_oracle", "c_estimated"]:
        prob_dir = RESULTS_DIR / variant
        model = EivToyModel()
        dfs = build_petab_dfs(
            variant=variant,
            cond_ids=cond_ids,
            c_true=c_true,
            c_obs=c_obs,
            t_obs=t_obs,
            y_obs=y_obs,
            sigma_y=sigma_y,
            sigma_c_prior=sigma_c,
        )
        write_problem_dir(prob_dir, model, dfs)
        out = fit_variant(prob_dir, model, n_starts=20)

        k_hat = out["x"]["k"]
        summary_rows.append(
            {
                "variant": variant,
                "k_hat": round(k_hat, 4),
                "k_true": k_true,
                "abs_err": round(k_hat - k_true, 4),
                "rel_err_%": round(100 * (k_hat - k_true) / k_true, 2),
                "fval": round(out["fval"], 4),
            }
        )

    print()
    print("=" * 60)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
