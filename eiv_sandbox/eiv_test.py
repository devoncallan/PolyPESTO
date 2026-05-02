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
            "k_rate": pet.FitParameter(
                id="k_rate",
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
    # NOTE: AMICI treats single-letter ids `k`, `y`, `t`, `p`, `x`, `w`, `h`
    # as reserved and silently prefixes them with `amici_`, which breaks
    # pypesto's parameter mapping. We use multi-char ids to avoid this.
    sbml.create_parameter(model, "k_rate", value=1.0, constant=False)
    sbml.create_parameter(model, "c", value=1.0, constant=True)
    sbml.create_species(model, "y", initialAmount=0.0)
    sbml.create_rate_rule(model, "y", "k_rate * c")
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
        "k_rate": pet.FitParameter(
            id="k_rate",
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


def fit_variant(
    prob_dir: Path,
    model: ModelBase,
    n_starts: int = 10,
    profile_k: bool = False,
    n_samples: int = 0,
) -> Dict[str, object]:
    """Optimize, optionally profile k_rate, and optionally run MCMC.

    Joint-MAP / profile-likelihood and marginal MCMC report different things
    for the EIV variant -- this function returns both so they can be compared.
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

    out: Dict[str, object] = {
        "fval": float(best.fval),
        "x": x_dict,
        "profile": None,
        "samples": None,
    }

    if profile_k:
        import pypesto.profile as profile  # type: ignore

        k_idx = x_names.index("k_rate")
        result = profile.parameter_profile(
            problem=prob.pypesto_problem,
            result=result,
            optimizer=optimizer,
            profile_index=np.array([k_idx]),
            engine=engine.SingleCoreEngine(),
            progress_bar=False,
        )
        prof = result.profile_result.list[0][k_idx]
        k_path = np.asarray(prof.x_path[k_idx])
        f_path = np.asarray(prof.fval_path)
        out["profile"] = {"k": k_path, "fval": f_path}

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
            x0=np.asarray(best.x),
        )
        # Cold chain only.
        trace = np.asarray(sample_result.sample_result.trace_x)[0]
        burn = n_samples // 2
        post = trace[burn:]
        out["samples"] = {name: post[:, i] for i, name in enumerate(x_names)}

    return out

    return out


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

    fits: Dict[str, Dict[str, object]] = {}
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
        out = fit_variant(
            prob_dir, model,
            n_starts=500,
            profile_k=True,
            n_samples=10000,
        )
        fits[variant] = out

        k_hat = out["x"]["k_rate"]
        row = {
            "variant": variant,
            "k_hat": round(k_hat, 4),
            "k_true": k_true,
            "abs_err": round(k_hat - k_true, 4),
            "rel_err_%": round(100 * (k_hat - k_true) / k_true, 2),
            "fval": round(out["fval"], 4),
        }
        if out["profile"] is not None:
            k_path = out["profile"]["k"]
            f_path = out["profile"]["fval"]
            df = f_path - f_path.min()
            mask = df <= 0.5
            if mask.any():
                lo = float(np.min(k_path[mask]))
                hi = float(np.max(k_path[mask]))
                row["prof_CI"] = f"[{lo:.3f}, {hi:.3f}]"
                row["prof_width"] = round(hi - lo, 4)
        if out["samples"] is not None:
            k_samp = out["samples"]["k_rate"]
            row["mcmc_mean"] = round(float(np.mean(k_samp)), 4)
            row["mcmc_std"] = round(float(np.std(k_samp)), 4)
            mlo, mhi = np.percentile(k_samp, [2.5, 97.5])
            row["mcmc_95_CI"] = f"[{mlo:.3f}, {mhi:.3f}]"
        summary_rows.append(row)

        if variant == "c_estimated":
            c_hat = [out["x"][f"c_cond{i + 1}"] for i in range(c_true.size)]
            print(f"\nc_estimated fit details:")
            print(f"  c_true   = {[round(v, 4) for v in c_true.tolist()]}")
            print(f"  c_obs    = {[round(v, 4) for v in c_obs.tolist()]}")
            print(f"  c_hat    = {[round(v, 4) for v in c_hat]}")

    print()
    print("=" * 80)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("=" * 80)

    # ----- Plots -----
    plot_fit(
        c_true, c_obs, t_obs, y_obs, fits, k_true,
        out_path=RESULTS_DIR / "fit.png",
    )
    plot_profile_k(
        fits, k_true, out_path=RESULTS_DIR / "profile_k.png",
    )
    plot_posterior_k(
        fits, k_true, out_path=RESULTS_DIR / "posterior_k.png",
    )
    print(f"\nSaved fit plot:       {RESULTS_DIR / 'fit.png'}")
    print(f"Saved profile plot:   {RESULTS_DIR / 'profile_k.png'}")
    print(f"Saved posterior plot: {RESULTS_DIR / 'posterior_k.png'}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_fit(
    c_true: np.ndarray,
    c_obs: np.ndarray,
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    fits: Dict[str, Dict[str, object]],
    k_true: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cond = c_true.size
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    t_dense = np.linspace(0, 1, 50)
    colors = {"c_measured": "tab:red", "c_oracle": "tab:green", "c_estimated": "tab:blue"}

    for i, ax in enumerate(axes.flat):
        if i >= n_cond:
            ax.axis("off")
            continue
        ax.scatter(t_obs, y_obs[i], color="black", zorder=5, label="data")
        # Truth
        ax.plot(t_dense, k_true * c_true[i] * t_dense, "k--", lw=1, alpha=0.5, label="truth")
        # Each variant's MAP curve
        for variant, out in fits.items():
            k_hat = out["x"]["k_rate"]
            if variant == "c_estimated":
                c_used = out["x"][f"c_cond{i + 1}"]
            elif variant == "c_oracle":
                c_used = c_true[i]
            else:  # c_measured
                c_used = c_obs[i]
            ax.plot(t_dense, k_hat * c_used * t_dense, color=colors[variant], lw=1.5, label=variant)
        ax.set_title(f"cond{i+1} (c_true={c_true[i]:.2f}, c_obs={c_obs[i]:.2f})", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Fit comparison (k_true={k_true})", fontsize=11)
    fig.supxlabel("time")
    fig.supylabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_posterior_k(
    fits: Dict[str, Dict[str, object]],
    k_true: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"c_measured": "tab:red", "c_oracle": "tab:green", "c_estimated": "tab:blue"}
    for variant, out in fits.items():
        if out["samples"] is None:
            continue
        k_samp = out["samples"]["k_rate"]
        ax.hist(
            k_samp, bins=50, density=True, histtype="step", lw=2,
            color=colors[variant],
            label=f"{variant}  (mean={np.mean(k_samp):.3f}, std={np.std(k_samp):.3f})",
        )
    ax.axvline(k_true, color="black", linestyle="--", lw=1, label=f"k_true={k_true}")
    ax.set_xlabel("k_rate")
    ax.set_ylabel("marginal posterior density")
    ax.set_title("MCMC marginal posterior on k_rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_profile_k(
    fits: Dict[str, Dict[str, object]],
    k_true: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"c_measured": "tab:red", "c_oracle": "tab:green", "c_estimated": "tab:blue"}
    for variant, out in fits.items():
        if out["profile"] is None:
            continue
        k_path = out["profile"]["k"]
        f_path = out["profile"]["fval"]
        df = f_path - f_path.min()
        order = np.argsort(k_path)
        ax.plot(k_path[order], df[order], color=colors[variant], lw=1.6, marker="o", ms=3,
                label=variant)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1, label=r"$\Delta\,$NLL = 0.5  (1$\sigma$)")
    ax.axhline(2.0, color="gray", linestyle="-.", lw=1, label=r"$\Delta\,$NLL = 2.0  (2$\sigma$)")
    ax.axvline(k_true, color="black", linestyle="--", lw=1, label=f"k_true={k_true}")
    ax.set_xlabel("k_rate")
    ax.set_ylabel(r"profile $-\log\,p\;(k)\;-\;$min")
    ax.set_title("Profile likelihood for k_rate")
    ax.set_ylim(-0.05, 4.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
