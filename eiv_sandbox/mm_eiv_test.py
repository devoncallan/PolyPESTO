"""Michaelis-Menten EIV test for polypesto.

ODE: dS/dt = -Vmax * S / (Km + S),  S(0) = S0.
Observable: S(t).

Two global parameters (Vmax, Km) are estimated. Per-condition initial
substrate S0 is the EIV variable: it is "set" experimentally (pipetting)
and "measured" with noise. Three variants compare:

  1. S0_measured  -- S0 fixed in condition table at noisy measurement.
  2. S0_oracle    -- S0 fixed at true value (oracle).
  3. S0_estimated -- S0 per-condition fit with normal prior centered on
                     the measurement (the EIV setup).

Compared to the simple `dy/dt = k*c` toy, this exercises:
  * parameter correlation (Vmax-Km posteriors are anti-correlated)
  * non-linear ODE dynamics
  * EIV on an *initial condition* rather than a rate scale.

Run:  python eiv_sandbox/mm_eiv_test.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import petab.v1.C as PC
from scipy.integrate import odeint

from polypesto.core import Problem
from polypesto.core import petab as pet
from polypesto.models import ModelBase, sbml

SANDBOX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SANDBOX_DIR / "_mm_results"


# ---------------------------------------------------------------------------
# Toy MM model
# ---------------------------------------------------------------------------


class MmEivModel(ModelBase):
    def _default_obs(self) -> List[str]:
        return ["S"]

    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
        # Placeholder; actual parameter table is written per-variant.
        return {
            "Vmax": pet.FitParameter(
                id="Vmax", scale=PC.LOG10, bounds=(1e-2, 1e2),
                nominal_value=1.0, estimate=True,
            ),
            "Km": pet.FitParameter(
                id="Km", scale=PC.LOG10, bounds=(1e-3, 1e2),
                nominal_value=0.5, estimate=True,
            ),
        }

    def _default_sbml_model(self) -> sbml.ModelDefinition:
        return _mm_sbml()

    def _default_solver_options(self, solver):
        solver.setRelativeTolerance(1e-9)
        solver.setAbsoluteTolerance(1e-11)
        solver.setSensitivityMethod(1)
        solver.setSensitivityOrder(1)
        return solver


def _mm_sbml() -> sbml.ModelDefinition:
    document, model = sbml.init_model("mm_eiv")
    sbml.create_compartment(model, "env", spatialDimensions=0, units="dimensionless")
    sbml.create_parameter(model, "Vmax", value=1.0, constant=False)
    sbml.create_parameter(model, "Km", value=0.5, constant=False)
    sbml.create_parameter(model, "S0", value=1.0, constant=True)  # condition override
    sbml.create_species(model, "S", initialAmount=0.0)
    sbml.create_initial_assignment(model, "S", "S0")
    sbml.create_rate_rule(model, "S", "-Vmax * S / (Km + S)")
    return sbml.create_model(model, document)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _mm_rhs(S: float, t: float, Vmax: float, Km: float) -> float:
    return -Vmax * S / (Km + S + 1e-12)


def simulate_progress(Vmax: float, Km: float, S0: float, t_obs: np.ndarray) -> np.ndarray:
    sol = odeint(_mm_rhs, S0, np.concatenate([[0.0], t_obs]), args=(Vmax, Km), rtol=1e-10, atol=1e-12)
    return sol[1:, 0]


def generate_data(
    Vmax_true: float,
    Km_true: float,
    S0_true: np.ndarray,
    t_obs: np.ndarray,
    sigma_y: float,
    sigma_S0_rel: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_cond = S0_true.size
    y_true = np.stack([
        simulate_progress(Vmax_true, Km_true, S0_true[i], t_obs)
        for i in range(n_cond)
    ])
    y_obs = y_true + rng.normal(0.0, sigma_y, size=y_true.shape)
    S0_obs = S0_true + rng.normal(0.0, sigma_S0_rel * S0_true, size=S0_true.shape)
    # Clip to positive; pipetting noise can't go negative
    S0_obs = np.clip(S0_obs, 1e-3, None)
    return y_obs, S0_obs


# ---------------------------------------------------------------------------
# PEtab assembly per variant
# ---------------------------------------------------------------------------


def build_petab_dfs(
    variant: str,
    cond_ids: List[str],
    S0_true: np.ndarray,
    S0_obs: np.ndarray,
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    sigma_y: float,
    sigma_S0_rel: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs_df = pet.utils.obs.define({"S": "S"})

    cond_rows: List[Dict[str, object]] = []
    for i, cid in enumerate(cond_ids):
        if variant == "S0_estimated":
            cval: object = f"S0_{cid}"
        elif variant == "S0_oracle":
            cval = float(S0_true[i])
        elif variant == "S0_measured":
            cval = float(S0_obs[i])
        else:
            raise ValueError(variant)
        cond_rows.append(
            {PC.CONDITION_ID: cid, PC.CONDITION_NAME: cid, "S0": cval}
        )
    cond_df = pet.utils.cond.format(pd.DataFrame(cond_rows))

    params: Dict[str, pet.FitParameter] = {
        "Vmax": pet.FitParameter(
            id="Vmax", scale=PC.LOG10, bounds=(1e-2, 1e2),
            nominal_value=1.0, estimate=True,
        ),
        "Km": pet.FitParameter(
            id="Km", scale=PC.LOG10, bounds=(1e-3, 1e2),
            nominal_value=0.5, estimate=True,
        ),
    }
    if variant == "S0_estimated":
        for i, cid in enumerate(cond_ids):
            mu = float(S0_obs[i])
            sd = sigma_S0_rel * mu
            params[f"S0_{cid}"] = pet.FitParameter(
                id=f"S0_{cid}",
                scale=PC.LIN,
                bounds=(1e-4, 1e2),
                nominal_value=mu,
                estimate=True,
                prior_type="normal",
                prior_params=f"{mu};{sd}",
            )
    param_df = pet.utils.param.define(params)

    obs_id = "obs_S"  # polypesto's ID.obs_id("S")
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
# Optimize + sample
# ---------------------------------------------------------------------------


def fit_variant(
    prob_dir: Path,
    model: ModelBase,
    n_starts: int = 500,
    n_samples: int = 10000,
) -> Dict[str, object]:
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
    x_full = np.asarray(best.x)

    # Unscale log10 parameters back to linear values for reporting
    scales = list(prob.pypesto_problem.x_scales)
    x_lin = {}
    for name, val, scale in zip(x_names, x_full, scales):
        x_lin[name] = float(10.0 ** val) if scale == PC.LOG10 else float(val)

    out: Dict[str, object] = {
        "fval": float(best.fval),
        "x_scaled": dict(zip(x_names, x_full.tolist())),
        "x_lin": x_lin,
        "x_scales": dict(zip(x_names, scales)),
        "samples_scaled": None,
        "samples_lin": None,
    }

    if n_samples > 0:
        import pypesto.sample as sample  # type: ignore

        sampler = sample.AdaptiveParallelTemperingSampler(
            internal_sampler=sample.AdaptiveMetropolisSampler(),
            n_chains=3,
        )
        sample_result = sample.sample(
            problem=prob.pypesto_problem,
            sampler=sampler,
            n_samples=n_samples,
            x0=x_full,
        )
        # Cold chain only.
        trace = np.asarray(sample_result.sample_result.trace_x)[0]
        burn = n_samples // 2
        post_scaled = trace[burn:]
        out["samples_scaled"] = {name: post_scaled[:, i] for i, name in enumerate(x_names)}
        # Unscale samples to linear values
        post_lin = {}
        for i, name in enumerate(x_names):
            col = post_scaled[:, i]
            post_lin[name] = 10.0 ** col if scales[i] == PC.LOG10 else col
        out["samples_lin"] = post_lin

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    seed = 7
    Vmax_true = 1.0
    Km_true = 0.5
    # 8 conditions log-spaced from 0.3*Km to 4*Km --- covers both linear
    # (S << Km) and saturating (S >> Km) regimes, required to identify
    # both Vmax and Km separately.
    S0_true = np.geomspace(0.15, 2.0, 8)
    t_obs = np.linspace(0.05, 1.0, 6)
    sigma_y = 0.03                  # ~3% absolute on S
    sigma_S0_rel = 0.10             # 10% relative pipetting error on S0

    cond_ids = [f"cond{i + 1}" for i in range(S0_true.size)]
    y_obs, S0_obs = generate_data(
        Vmax_true, Km_true, S0_true, t_obs, sigma_y, sigma_S0_rel, seed=seed
    )

    print(f"Vmax_true = {Vmax_true},  Km_true = {Km_true}")
    print(f"S0_true   = {[round(v, 4) for v in S0_true.tolist()]}")
    print(f"S0_obs    = {[round(v, 4) for v in S0_obs.tolist()]}")
    print(f"sigma_y   = {sigma_y},  sigma_S0/S0 = {sigma_S0_rel:.0%}")
    print()

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)

    fits: Dict[str, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []
    for variant in ["S0_measured", "S0_oracle", "S0_estimated"]:
        prob_dir = RESULTS_DIR / variant
        model = MmEivModel()
        dfs = build_petab_dfs(
            variant=variant,
            cond_ids=cond_ids,
            S0_true=S0_true,
            S0_obs=S0_obs,
            t_obs=t_obs,
            y_obs=y_obs,
            sigma_y=sigma_y,
            sigma_S0_rel=sigma_S0_rel,
        )
        write_problem_dir(prob_dir, model, dfs)
        out = fit_variant(prob_dir, model, n_starts=500, n_samples=10000)
        fits[variant] = out

        Vmax_hat = out["x_lin"]["Vmax"]
        Km_hat = out["x_lin"]["Km"]
        row = {
            "variant": variant,
            "Vmax_hat": round(Vmax_hat, 4),
            "Km_hat": round(Km_hat, 4),
            "fval": round(out["fval"], 4),
        }
        if out["samples_lin"] is not None:
            for p_name, p_true in [("Vmax", Vmax_true), ("Km", Km_true)]:
                samp = out["samples_lin"][p_name]
                row[f"{p_name}_mean"] = round(float(np.mean(samp)), 4)
                row[f"{p_name}_std"] = round(float(np.std(samp)), 4)
                lo, hi = np.percentile(samp, [2.5, 97.5])
                row[f"{p_name}_95CI"] = f"[{lo:.3f}, {hi:.3f}]"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print()
    print("=" * 110)
    print(df_summary.to_string(index=False))
    print("=" * 110)
    print(f"\n(truth: Vmax={Vmax_true}, Km={Km_true})")

    # ----- Plots -----
    plot_fit(
        S0_true, S0_obs, t_obs, y_obs, fits, Vmax_true, Km_true,
        out_path=RESULTS_DIR / "fit.png",
    )
    plot_marginals(
        fits, Vmax_true, Km_true,
        out_path=RESULTS_DIR / "marginals.png",
    )
    plot_joint(
        fits, Vmax_true, Km_true,
        out_path=RESULTS_DIR / "joint.png",
    )
    print(f"\nSaved fit plot:       {RESULTS_DIR / 'fit.png'}")
    print(f"Saved marginal plot:  {RESULTS_DIR / 'marginals.png'}")
    print(f"Saved joint plot:     {RESULTS_DIR / 'joint.png'}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_fit(
    S0_true: np.ndarray,
    S0_obs: np.ndarray,
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    fits: Dict[str, Dict[str, object]],
    Vmax_true: float,
    Km_true: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cond = S0_true.size
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    t_dense = np.linspace(0, t_obs.max() * 1.05, 80)
    colors = {"S0_measured": "tab:red", "S0_oracle": "tab:green", "S0_estimated": "tab:blue"}

    for i, ax in enumerate(axes.flat):
        if i >= n_cond:
            ax.axis("off")
            continue
        ax.scatter(t_obs, y_obs[i], color="black", zorder=5, s=22, label="data")
        # Truth
        y_truth = simulate_progress(Vmax_true, Km_true, S0_true[i], t_dense)
        ax.plot(t_dense, y_truth, "k--", lw=1, alpha=0.5, label="truth")
        # Each variant's MAP curve
        for variant, out in fits.items():
            Vmax_hat = out["x_lin"]["Vmax"]
            Km_hat = out["x_lin"]["Km"]
            if variant == "S0_estimated":
                S0_used = out["x_lin"][f"S0_cond{i + 1}"]
            elif variant == "S0_oracle":
                S0_used = float(S0_true[i])
            else:
                S0_used = float(S0_obs[i])
            y_curve = simulate_progress(Vmax_hat, Km_hat, S0_used, t_dense)
            ax.plot(t_dense, y_curve, color=colors[variant], lw=1.4, label=variant)
        ax.set_title(f"cond{i+1}  S0_t={S0_true[i]:.2f}, S0_o={S0_obs[i]:.2f}", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Fit: Vmax_true={Vmax_true}, Km_true={Km_true}", fontsize=11)
    fig.supxlabel("time")
    fig.supylabel("S")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_marginals(
    fits: Dict[str, Dict[str, object]],
    Vmax_true: float,
    Km_true: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"S0_measured": "tab:red", "S0_oracle": "tab:green", "S0_estimated": "tab:blue"}
    for ax, p_name, p_true in zip(axes, ["Vmax", "Km"], [Vmax_true, Km_true]):
        for variant, out in fits.items():
            if out["samples_lin"] is None:
                continue
            samp = out["samples_lin"][p_name]
            ax.hist(
                samp, bins=60, density=True, histtype="step", lw=2,
                color=colors[variant],
                label=f"{variant}  (mean={np.mean(samp):.3f}, std={np.std(samp):.3f})",
            )
        ax.axvline(p_true, color="black", linestyle="--", lw=1, label=f"{p_name}_true={p_true}")
        ax.set_xlabel(p_name)
        ax.set_ylabel("posterior density")
        ax.set_title(f"Marginal posterior on {p_name}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_joint(
    fits: Dict[str, Dict[str, object]],
    Vmax_true: float,
    Km_true: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    colors = {"S0_measured": "tab:red", "S0_oracle": "tab:green", "S0_estimated": "tab:blue"}
    for ax, (variant, out) in zip(axes, fits.items()):
        if out["samples_lin"] is None:
            ax.axis("off")
            continue
        Vmax_s = out["samples_lin"]["Vmax"]
        Km_s = out["samples_lin"]["Km"]
        ax.scatter(Vmax_s, Km_s, s=2, alpha=0.15, color=colors[variant])
        ax.scatter([Vmax_true], [Km_true], marker="*", s=180,
                   color="black", edgecolor="white", lw=1, zorder=5, label="truth")
        ax.set_xlabel("Vmax")
        if ax is axes[0]:
            ax.set_ylabel("Km")
        corr = np.corrcoef(Vmax_s, Km_s)[0, 1]
        ax.set_title(f"{variant}  (corr={corr:+.2f})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Joint posterior  (Vmax, Km)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
