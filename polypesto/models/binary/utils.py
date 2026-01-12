from copy import deepcopy
from typing import Dict, List
from pathlib import Path

import numpy as np

from polypesto.core import (
    Experiment,
    Dataset,
    SimulatedProblem,
    create_sim_conditions,
    Problem,
    write_empty_problem,
)

from .irreversible import BinaryIrreversible
from .reversible import BinaryReversible


def set_time_as_conv(exp: Experiment) -> Experiment:

    datasets = exp.data
    cond = exp.conds.to_dict()


def expand_conds(conds: Dict[str, float]) -> Experiment:

    cond_names = conds.keys()

    if set(["A0", "B0"]).issubset(cond_names):
        conds["M0"] = conds["A0"] + conds["B0"]
        conds["fA0"] = conds["A0"] / conds["M0"]
        return conds
    elif set(["fA0", "cM0"]).issubset(cond_names):
        conds["A0"] = conds["fA0"] * conds["cM0"]
        conds["B0"] = (1 - conds["fA0"]) * conds["cM0"]
        return conds
    else:
        raise ValueError(
            f'Conditions must include either ("A0", "B0") or ("fA0", "cM0"). Actual: {list(cond_names)}'
        )


from copy import deepcopy
import numpy as np


def modify_dataset(
    ds: Dataset,
    fA0: float,
    conv_spread_95: float = 0.03,
    n_mc: int = 5000,
    seed: int = 42,
) -> Dataset:
    new_ds = deepcopy(ds)

    if set(["xA", "xB"]).issubset(ds.obs_map.keys()):
        # ---- deterministic transforms (your code) ----
        new_ds.data["x"] = fA0 * ds.data["xA"] + (1 - fA0) * ds.data["xB"]
        new_ds.data[ds.tkey] = new_ds.data["x"]

        mon_A = fA0 * (1 - new_ds.data["xA"])
        mon_B = (1 - fA0) * (1 - new_ds.data["xB"])
        new_ds.data["fA"] = mon_A / (mon_A + mon_B)
        new_ds.data["fB"] = mon_B / (mon_A + mon_B)

        new_ds.data["FA"] = (fA0 - (1 - new_ds.data["x"]) * new_ds.data["fA"]) / (
            new_ds.data["x"] + 1e-10
        )
        new_ds.data["FB"] = 1 - new_ds.data["FA"]

        new_ds.obs_map["fA"] = "fA"
        new_ds.obs_map["fB"] = "fB"
        new_ds.obs_map["FA"] = "FA"
        new_ds.obs_map["FB"] = "FB"
        new_ds.obs_map["x"] = "x"

        # ---- MC propagation from measured xA, xB ----
        z = 1.96
        sigma_x = conv_spread_95 / z  # 1σ for xA and xB (absolute in conversion units)

        xA = new_ds.data["xA"].to_numpy(dtype=float)
        xB = new_ds.data["xB"].to_numpy(dtype=float)
        T = xA.size

        rng = np.random.default_rng(seed)

        if "xA" in new_ds.noise_map and isinstance(new_ds.noise_map["xA"], float):
            sigma_xA = float(new_ds.noise_map["xA"])
        else:
            print("WARNING: No noise_map for xA, setting to 0...")
            sigma_xA = 0.0
        if "xB" in new_ds.noise_map and isinstance(new_ds.noise_map["xB"], float):
            sigma_xB = float(new_ds.noise_map["xB"])
        else:
            print("WARNING: No noise_map for xB, setting to 0...")
            sigma_xB = 0.0

        xA_s = np.clip(rng.normal(xA, sigma_xA, size=(n_mc, T)), 0.0, 1.0)
        xB_s = np.clip(rng.normal(xB, sigma_xB, size=(n_mc, T)), 0.0, 1.0)

        x_s = fA0 * xA_s + (1 - fA0) * xB_s

        monA_s = fA0 * (1 - xA_s)
        monB_s = (1 - fA0) * (1 - xB_s)
        
        fA_s = np.clip(monA_s / (monA_s + monB_s), 0.0, 1.0)
        FA_s = np.clip((fA0 - (1 - x_s) * fA_s) / (x_s + 1e-10), 0.0, 1.0)

        # 1σ per timepoint (what you want for PEtab/pyPESTO normal noise)
        new_ds.data["x_sigma"] = np.std(x_s, axis=0, ddof=1)
        new_ds.data["fA_sigma"] = np.std(fA_s, axis=0, ddof=1)
        new_ds.data["FA_sigma"] = np.std(FA_s, axis=0, ddof=1)

        # fB = 1 - fA, FB = 1 - FA -> same sigma
        new_ds.data["fB_sigma"] = new_ds.data["fA_sigma"]
        new_ds.data["FB_sigma"] = new_ds.data["FA_sigma"]

        # Point noise_map to the sigma columns (per-point noise)
        new_ds.noise_map["x"] = "x_sigma"
        new_ds.noise_map["fA"] = "fA_sigma"
        new_ds.noise_map["fB"] = "fB_sigma"
        new_ds.noise_map["FA"] = "FA_sigma"
        new_ds.noise_map["FB"] = "FB_sigma"

    elif set(["x", "fA"]).issubset(ds.obs_map.keys()):
        pass
    elif set(["nA", "nB"]).issubset(ds.obs_map.keys()):
        pass
    else:
        raise ValueError(
            f'Dataset observable map must include either ("xA", "xB") or ("x", "fA"). Actual: {list(ds.obs_map.keys())}'
        )

    return new_ds


# def modify_dataset(ds: Dataset, fA0: float) -> Dataset:

#     # Check obs_map has

#     # new_data = ds.data.copy()
#     new_ds = deepcopy(ds)
#     if set(["xA", "xB"]).issubset(ds.obs_map.keys()):

#         # Set independent variable to conversion
#         new_ds.data["x"] = fA0 * ds.data["xA"] + (1 - fA0) * ds.data["xB"]
#         new_ds.data[ds.tkey] = new_ds.data["x"]

#         mon_A = fA0 * (1 - new_ds.data["xA"])
#         mon_B = (1 - fA0) * (1 - new_ds.data["xB"])
#         new_ds.data["fA"] = mon_A / (mon_A + mon_B)
#         new_ds.data["fB"] = mon_B / (mon_A + mon_B)

#         new_ds.data["FA"] = (fA0 - (1 - new_ds.data["x"]) * new_ds.data["fA"]) / (
#             new_ds.data["x"] + 1e-10
#         )
#         new_ds.data["FB"] = 1 - new_ds.data["FA"]

#         new_ds.obs_map["fA"] = "fA"
#         new_ds.obs_map["fB"] = "fB"
#         new_ds.obs_map["FA"] = "FA"
#         new_ds.obs_map["FB"] = "FB"

#         new_ds.noise_map["fA"] = 0.03
#         new_ds.noise_map["fB"] = 0.03
#         new_ds.noise_map["FA"] = 0.03
#         new_ds.noise_map["FB"] = 0.03

#     elif set(["x", "fA"]).issubset(ds.obs_map.keys()):
#         pass
#         # new_data[ds.tkey] = ds.data[ds.obs_map["x"]]
#     elif set(["nA", "nB"]).issubset(ds.obs_map.keys()):
#         pass
#     else:
#         raise ValueError(
#             f'Dataset observable map must include either ("xA", "xB") or ("x", "fA"). Actual: {list(ds.obs_map.keys())}'
#         )

#     return new_ds


def modify_experiments(
    experiments: List[Experiment], fA0s: List[float] | None = None, **kwargs
) -> List[Experiment]:

    # Convert tkey in data to conversion
    # Add fA and fB to obs

    new_exps = []
    for exp in experiments:

        datasets = exp.data
        cond = exp.conds.to_dict()
        cond = expand_conds(cond)

        fA0 = cond["fA0"]
        fB0 = 1 - fA0

        if fA0s is not None and not np.isclose(fA0, fA0s, atol=1e-3).any():
            print(f"Skipping experiment {exp.id} with fA0={fA0} not in {fA0s}")
            continue

        new_datasets = []
        for ds in datasets:
            new_ds = modify_dataset(ds, fA0, **kwargs)
            new_datasets.append(new_ds)

        exp = Experiment(
            id=exp.id,
            conds=exp.conds,
            data=new_datasets,
        )
        new_exps.append(exp)

    return new_exps


# def modify_experiments(experiments: List[Experiment]) -> List[Experiment]:

#     # Convert tkey in data to conversion
#     # Add fA and fB to obs

#     new_exps = []
#     for exp in experiments:

#         datasets = exp.data
#         cond = exp.conds.to_dict()
#         cond = expand_conds(cond)

#         fA0 = cond["fA0"]
#         fB0 = 1 - fA0

#         new_datasets = []
#         for ds in datasets:

#             assert ds.tkey in ds.data.columns
#             assert "xA" in ds.obs_map and "xB" in ds.obs_map

#             new_data = ds.data.copy()
#             new_data[ds.tkey] = (
#                 fA0 * ds.data[ds.obs_map["xA"]] + fB0 * ds.data[ds.obs_map["xB"]]
#             )

#             mon_A = fA0 * (1 - new_data[ds.obs_map["xA"]])
#             mon_B = fB0 * (1 - new_data[ds.obs_map["xB"]])

#             new_data["fA"] = mon_A / (mon_A + mon_B)
#             new_data["fB"] = mon_B / (mon_A + mon_B)

#             new_data["FA"] = (fA0 - (1 - new_data[ds.tkey]) * new_data["fA"]) / (
#                 new_data[ds.tkey] + 1e-10
#             )
#             new_data["FB"] = 1 - new_data["FA"]

#             # ds.obs_map["fA"] = "fA"
#             # ds.obs_map["fB"] = "fB"

#             # if ds.noise_map is not None:
#             #     ds.noise_map["fA"] = ds.noise_map.get("xA", 0.0) * fA0
#             #     ds.noise_map["fB"] = ds.noise_map.get("xB", 0.0) * fB0

#             ds.data = new_data

#             new_datasets.append(ds)

#         exp = Experiment(
#             id=exp.id,
#             conds=exp.conds,
#             data=new_datasets,
#         )
#         new_exps.append(exp)

#     return new_exps


def create_ensemble_pred_problem(
    data_dir: Path | str, model: BinaryIrreversible | BinaryReversible
) -> SimulatedProblem:

    fA0s = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    cM0s = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    A0s = fA0s * cM0s
    B0s = (1 - fA0s) * cM0s

    problem = write_empty_problem(
        prob_dir=data_dir,
        model=model,
        sim_conds=create_sim_conditions(
            true_params={},
            t_evals=np.arange(0.01, 0.9, 0.01),
            conds=dict(
                A0=A0s,
                B0=B0s,
            ),
        ),
    )

    return problem


def create_condition_grid_pred_problem(
    prob: Problem,
    n_points: int = 50,
    output_dir: Path | None = None,
) -> SimulatedProblem:
    """
    Build a prediction problem with per-condition dense evaluation grids.

    For each condition present in the measurement table, simulate from the
    earliest to latest "time" (or conversion) with `n_points` evenly spaced
    points. Reuses the condition values (e.g., A0/B0) from the PEtab condition
    table and writes an empty PEtab problem suitable for ensemble prediction.
    """

    import pandas as pd
    import petab.v1.C as C

    mdf = prob.petab_problem.measurement_df
    cond_df = prob.petab_problem.condition_df

    # Condition IDs from measurements; align on conditionId in condition_df
    cond_ids = mdf[C.SIMULATION_CONDITION_ID].unique().tolist()
    if not cond_ids:
        raise ValueError("No conditions found in measurement_df.")

    # Per-condition grids and condition values (numeric only)
    t_evals: list[np.ndarray] = []
    cond_rows: list[dict] = []
    for cond_id in cond_ids:
        sub = mdf[mdf[C.SIMULATION_CONDITION_ID] == cond_id]
        t_vals = pd.to_numeric(sub[C.TIME], errors="coerce").dropna().to_numpy()
        if t_vals.size == 0:
            raise ValueError(f"No time values for condition '{cond_id}'.")
        t_min, t_max = float(t_vals.min()), float(t_vals.max())
        t_evals.append(np.linspace(t_min, t_max, n_points))

        if cond_id not in cond_df.index:
            raise KeyError(f"Condition '{cond_id}' not found in condition_df.")
        row = cond_df.loc[cond_id]
        numeric_row = {
            k: float(v) for k, v in row.items() if pd.notna(v) and isinstance(v, (int, float))
        }
        cond_rows.append(numeric_row)

    if not cond_rows:
        raise ValueError("No condition rows assembled from condition_df.")

    keys = list(cond_rows[0].keys())
    conds = {k: [r.get(k, np.nan) for r in cond_rows] for k in keys}

    sim_conds = create_sim_conditions(
        true_params={}, conds=conds, t_evals=t_evals, meas_noise=0.05
    )
    # Preserve original condition IDs so predictions align with measurement IDs
    if len(sim_conds) != len(cond_ids):
        raise ValueError(
            f"Mismatch between sim conditions ({len(sim_conds)}) and measurement condition IDs ({len(cond_ids)})."
        )
    for i, cid in enumerate(cond_ids):
        sim_conds[i].conds = sim_conds[i].conds.set_id(cid)

    pred_dir = output_dir or prob.paths.ensemble_dir / "grid_pred"
    pred_problem = write_empty_problem(pred_dir, prob.model, sim_conds)
    return pred_problem
