"""EIV test on the polypesto irreversible binary copolymerization model.

This is the realistic test: per-aliquot final conversion `xf` is the EIV
variable, observable `FA` (cumulative copolymer composition) is measured
at SBML time=1 (corresponding to actual conversion = xf for that aliquot).

Two global parameters: `rA`, `rB` (reactivity ratios). Per-aliquot `xf`
is "set" experimentally and "measured" via NMR with absolute noise sigma_xf.

Three variants:

  1. xf_measured  -- xf fixed in condition table at noisy NMR measurement.
  2. xf_oracle    -- xf fixed at true value (oracle).
  3. xf_estimated -- xf per-aliquot fit with normal prior centered at the
                     measurement (the EIV setup).

Run:  python eiv_sandbox/cpe_eiv_test.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import petab.v1.C as PC

from polypesto.core import Problem, create_sim_conditions, simulate_problem
from polypesto.core import petab as pet
from polypesto.models import ModelBase
from polypesto.models.binary import BinaryIrreversibleTime

SANDBOX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SANDBOX_DIR / "_cpe_results"


# ---------------------------------------------------------------------------
# Synthetic data via polypesto.simulate_problem
# ---------------------------------------------------------------------------


def generate_truth_data(
    rA_true: float,
    rB_true: float,
    A0: np.ndarray,
    B0: np.ndarray,
    xf_true: np.ndarray,
    sigma_FA: float,
    seed: int,
) -> np.ndarray:
    """Simulate `BinaryIrreversibleTime` at truth params for each condition,
    return noisy FA observations at SBML time=1."""
    truth_dir = SANDBOX_DIR / "_cpe_truth"
    if truth_dir.exists():
        shutil.rmtree(truth_dir)

    np.random.seed(seed)
    model = BinaryIrreversibleTime()
    sim_conds = create_sim_conditions(
        true_params={"rA": rA_true, "rB": rB_true, "rX": 1.0},
        conds={"A0": A0.tolist(), "B0": B0.tolist(), "xf": xf_true.tolist()},
        t_evals=np.array([1.0]),
        meas_noise=sigma_FA,
    )
    sim_problem = simulate_problem(
        prob_dir=str(truth_dir), model=model, conds=sim_conds, overwrite=True,
    )
    meas_df = sim_problem.petab_problem.measurement_df
    fa_obs = np.zeros(len(A0))
    for i, sc in enumerate(sim_problem.sim_conditions):
        cid = sc.conds.id
        rows = meas_df[meas_df[PC.SIMULATION_CONDITION_ID] == cid]
        fa_obs[i] = float(rows[PC.MEASUREMENT].values[0])
    return fa_obs


# ---------------------------------------------------------------------------
# PEtab assembly per variant
# ---------------------------------------------------------------------------


def build_petab_dfs(
    variant: str,
    cond_ids: List[str],
    A0: np.ndarray,
    B0: np.ndarray,
    xf_true: np.ndarray,
    xf_obs: np.ndarray,
    fa_obs: np.ndarray,
    sigma_FA: float,
    sigma_xf: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs_df = pet.utils.obs.define({"FA": "FA"})

    cond_rows: List[Dict[str, object]] = []
    for i, cid in enumerate(cond_ids):
        if variant == "xf_estimated":
            xf_val: object = f"xf_{cid}"
        elif variant == "xf_oracle":
            xf_val = float(xf_true[i])
        elif variant == "xf_measured":
            xf_val = float(xf_obs[i])
        else:
            raise ValueError(variant)
        cond_rows.append(
            {
                PC.CONDITION_ID: cid,
                PC.CONDITION_NAME: cid,
                "A0": float(A0[i]),
                "B0": float(B0[i]),
                "xf": xf_val,
            }
        )
    cond_df = pet.utils.cond.format(pd.DataFrame(cond_rows))

    params: Dict[str, pet.FitParameter] = {
        "rA": pet.FitParameter(
            id="rA", scale=PC.LOG10, bounds=(1e-2, 1e2),
            nominal_value=1.0, estimate=True,
        ),
        "rB": pet.FitParameter(
            id="rB", scale=PC.LOG10, bounds=(1e-2, 1e2),
            nominal_value=1.0, estimate=True,
        ),
        "rX": pet.FitParameter(
            id="rX", scale=PC.LOG10, bounds=(1e-2, 1e2),
            nominal_value=1.0, estimate=False,
        ),
    }
    if variant == "xf_estimated":
        for i, cid in enumerate(cond_ids):
            mu = float(xf_obs[i])
            params[f"xf_{cid}"] = pet.FitParameter(
                id=f"xf_{cid}",
                scale=PC.LIN,
                bounds=(1e-3, 1.0),
                nominal_value=mu,
                estimate=True,
                prior_type="normal",
                prior_params=f"{mu};{sigma_xf}",
            )
    param_df = pet.utils.param.define(params)

    obs_id = "obs_FA"
    rows: List[Dict[str, object]] = []
    for i, cid in enumerate(cond_ids):
        rows.append(
            {
                PC.OBSERVABLE_ID: obs_id,
                PC.SIMULATION_CONDITION_ID: cid,
                PC.TIME: 1.0,
                PC.MEASUREMENT: float(fa_obs[i]),
                PC.NOISE_PARAMETERS: float(sigma_FA),
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
    pp = prob.pypesto_problem
    free_idx = list(pp.x_free_indices)
    # pypesto's x_names and x_scales include fixed params too -- reduce to free.
    x_names = [pp.x_names[i] for i in free_idx]
    scales = [pp.x_scales[i] for i in free_idx]
    # best.x is the FULL parameter vector (incl. fixed params); reduce to estimated.
    x_full = np.asarray(best.x)
    x_reduced = pp.get_reduced_vector(x_full)

    x_lin = {}
    for name, val, scale in zip(x_names, x_reduced, scales):
        x_lin[name] = float(10.0 ** val) if scale == PC.LOG10 else float(val)

    out: Dict[str, object] = {
        "fval": float(best.fval),
        "x_scaled": dict(zip(x_names, x_reduced.tolist())),
        "x_lin": x_lin,
        "x_scales": dict(zip(x_names, scales)),
        "samples_lin": None,
    }

    if n_samples > 0:
        import pypesto.sample as sample  # type: ignore

        # warm_start_parallel_chains=1.0 disables prior-sample warm start,
        # which has a bug in pypesto 0.5.9 when problems have a mix of priored
        # and non-priored estimated parameters together with fixed parameters
        # (PriorStartpoints uses full-vector indices into a reduced array).
        sampler = sample.AdaptiveParallelTemperingSampler(
            internal_sampler=sample.AdaptiveMetropolisSampler(),
            n_chains=3,
            options={"warm_start_parallel_chains": 1.0},
        )
        sample_result = sample.sample(
            problem=prob.pypesto_problem,
            sampler=sampler,
            n_samples=n_samples,
            x0=x_reduced,
        )
        trace = np.asarray(sample_result.sample_result.trace_x)[0]
        burn = n_samples // 2
        post_scaled = trace[burn:]
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
    rA_true = 0.5
    rB_true = 2.0

    # 4 f0 values x 3 xf values = 12 aliquots. Each aliquot = 1 condition.
    f0_vals = np.array([0.25, 0.45, 0.65, 0.85])
    xf_vals = np.array([0.30, 0.55, 0.80])
    sigma_FA = 0.02   # NMR composition
    sigma_xf = 0.03   # NMR conversion

    # Build the (f0, xf) grid and corresponding (A0, B0, xf_true) arrays
    f0_grid, xf_grid = np.meshgrid(f0_vals, xf_vals, indexing="ij")
    f0_flat = f0_grid.flatten()
    xf_true = xf_grid.flatten()
    A0 = f0_flat                # cM0 = 1
    B0 = 1.0 - f0_flat
    n_cond = A0.size

    # Generate noisy data via polypesto's simulator
    fa_obs = generate_truth_data(
        rA_true, rB_true, A0, B0, xf_true, sigma_FA, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    xf_obs = np.clip(xf_true + rng.normal(0.0, sigma_xf, size=n_cond), 1e-3, 0.999)

    print(f"rA_true = {rA_true},  rB_true = {rB_true}")
    print(f"sigma_FA = {sigma_FA},  sigma_xf = {sigma_xf}")
    print(f"\nf0 / xf_true / xf_obs / FA_obs grid:")
    for i in range(n_cond):
        print(f"  cond{i+1:02d}  f0={f0_flat[i]:.2f}  xf_true={xf_true[i]:.2f}  "
              f"xf_obs={xf_obs[i]:.3f}  FA_obs={fa_obs[i]:.3f}")
    print()

    cond_ids = [f"cond{i+1:02d}" for i in range(n_cond)]

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)

    fits: Dict[str, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []
    for variant in ["xf_measured", "xf_oracle", "xf_estimated"]:
        prob_dir = RESULTS_DIR / variant
        model = BinaryIrreversibleTime()
        dfs = build_petab_dfs(
            variant=variant,
            cond_ids=cond_ids,
            A0=A0, B0=B0,
            xf_true=xf_true, xf_obs=xf_obs,
            fa_obs=fa_obs,
            sigma_FA=sigma_FA, sigma_xf=sigma_xf,
        )
        write_problem_dir(prob_dir, model, dfs)
        out = fit_variant(prob_dir, model, n_starts=500, n_samples=10000)
        fits[variant] = out

        rA_hat = out["x_lin"]["rA"]
        rB_hat = out["x_lin"]["rB"]
        row = {
            "variant": variant,
            "rA_hat": round(rA_hat, 4),
            "rB_hat": round(rB_hat, 4),
            "fval": round(out["fval"], 4),
        }
        if out["samples_lin"] is not None:
            for p_name, p_true in [("rA", rA_true), ("rB", rB_true)]:
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
    print(f"\n(truth: rA={rA_true}, rB={rB_true})")

    # ----- Plots -----
    plot_fit(
        f0_vals, f0_flat, xf_true, xf_obs, fa_obs, fits,
        rA_true, rB_true, sigma_FA, sigma_xf,
        out_path=RESULTS_DIR / "fit.png",
    )
    plot_marginals(
        fits, rA_true, rB_true,
        out_path=RESULTS_DIR / "marginals.png",
    )
    plot_joint(
        fits, rA_true, rB_true,
        out_path=RESULTS_DIR / "joint.png",
    )
    print(f"\nSaved fit plot:       {RESULTS_DIR / 'fit.png'}")
    print(f"Saved marginal plot:  {RESULTS_DIR / 'marginals.png'}")
    print(f"Saved joint plot:     {RESULTS_DIR / 'joint.png'}")


# ---------------------------------------------------------------------------
# FA(f0, xf, rA, rB) -- integrated copolymer composition
# Computed by simulating BinaryIrreversibleTime once per (f0, rA, rB) over
# a grid of xf values via reuse of the AMICI compile (fast).
# ---------------------------------------------------------------------------


def _fa_curve(
    f0: float, xf_grid: np.ndarray, rA: float, rB: float,
) -> np.ndarray:
    """Evaluate FA at SBML time=1 for several xf values at fixed f0,rA,rB.
    Uses a fresh tiny PEtab problem and amici simulation."""
    import os
    tmp_dir = SANDBOX_DIR / "_cpe_curvebuf"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    A0 = np.full_like(xf_grid, f0)
    B0 = np.full_like(xf_grid, 1.0 - f0)
    sim_conds = create_sim_conditions(
        true_params={"rA": rA, "rB": rB, "rX": 1.0},
        conds={"A0": A0.tolist(), "B0": B0.tolist(), "xf": xf_grid.tolist()},
        t_evals=np.array([1.0]),
        meas_noise=1e-6,  # AMICI requires sigma > 0; effectively zero noise.
    )
    sim_problem = simulate_problem(
        prob_dir=str(tmp_dir),
        model=BinaryIrreversibleTime(),
        conds=sim_conds,
        overwrite=True,
    )
    meas_df = sim_problem.petab_problem.measurement_df
    fa = np.zeros(xf_grid.size)
    for i, sc in enumerate(sim_problem.sim_conditions):
        cid = sc.conds.id
        rows = meas_df[meas_df[PC.SIMULATION_CONDITION_ID] == cid]
        fa[i] = float(rows[PC.MEASUREMENT].values[0])
    return fa


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_fit(
    f0_vals: np.ndarray,
    f0_flat: np.ndarray,
    xf_true: np.ndarray,
    xf_obs: np.ndarray,
    fa_obs: np.ndarray,
    fits: Dict[str, Dict[str, object]],
    rA_true: float,
    rB_true: float,
    sigma_FA: float,
    sigma_xf: float,
    out_path: Path,
) -> None:
    """For each f0, plot FA(xf) data with x and y error bars + truth and
    each variant's MAP curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(f0_vals), figsize=(4 * len(f0_vals), 4),
                             sharex=True, sharey=True)
    colors = {"xf_measured": "tab:red", "xf_oracle": "tab:green", "xf_estimated": "tab:blue"}
    xf_grid = np.linspace(0.05, 0.95, 25)

    for ax, f0 in zip(axes, f0_vals):
        # Truth curve
        fa_truth = _fa_curve(f0, xf_grid, rA_true, rB_true)
        ax.plot(xf_grid, fa_truth, "k--", lw=1, alpha=0.5, label="truth")

        # Each variant's MAP curve
        for variant, out in fits.items():
            rA_h = out["x_lin"]["rA"]
            rB_h = out["x_lin"]["rB"]
            fa_var = _fa_curve(f0, xf_grid, rA_h, rB_h)
            ax.plot(xf_grid, fa_var, color=colors[variant], lw=1.4, label=variant)

        # Data for this f0 (with both x and y error bars)
        mask = np.isclose(f0_flat, f0)
        ax.errorbar(
            xf_obs[mask], fa_obs[mask],
            xerr=sigma_xf, yerr=sigma_FA,
            fmt="o", color="black", ecolor="black",
            ms=5, capsize=3, lw=1, zorder=5,
            label="data $\\pm(\\sigma_{xf},\\sigma_{FA})$",
        )

        ax.set_title(f"f0 = {f0:.2f}", fontsize=10)
        ax.set_xlim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("FA")
            ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("xf")
    fig.suptitle(f"Cumulative composition  FA(xf) per f0  (rA_true={rA_true}, rB_true={rB_true})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_marginals(
    fits: Dict[str, Dict[str, object]],
    rA_true: float,
    rB_true: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"xf_measured": "tab:red", "xf_oracle": "tab:green", "xf_estimated": "tab:blue"}
    for ax, p_name, p_true in zip(axes, ["rA", "rB"], [rA_true, rB_true]):
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
    rA_true: float,
    rB_true: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    colors = {"xf_measured": "tab:red", "xf_oracle": "tab:green", "xf_estimated": "tab:blue"}
    for ax, (variant, out) in zip(axes, fits.items()):
        if out["samples_lin"] is None:
            ax.axis("off")
            continue
        rA_s = out["samples_lin"]["rA"]
        rB_s = out["samples_lin"]["rB"]
        ax.scatter(rA_s, rB_s, s=2, alpha=0.15, color=colors[variant])
        ax.scatter([rA_true], [rB_true], marker="*", s=180,
                   color="black", edgecolor="white", lw=1, zorder=5, label="truth")
        ax.set_xlabel("rA")
        if ax is axes[0]:
            ax.set_ylabel("rB")
        corr = np.corrcoef(rA_s, rB_s)[0, 1]
        ax.set_title(f"{variant}  (corr={corr:+.2f})", fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Joint posterior  (rA, rB)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
