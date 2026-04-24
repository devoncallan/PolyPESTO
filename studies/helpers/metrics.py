"""Quantitative recovery / identifiability metrics for Study results.

All helpers live outside ``polypesto/`` so we can iterate on analysis without
touching the core package.

Input: the DataFrame returned by ``Study.results_summary()`` (columns include
``true_value``, ``ensemble_median``, ``percentile_5/25/75/95``, ``lowerBound``,
``upperBound``, ``converged``). ``recovery_metrics`` adds derived columns and,
if given the ``Study``, chain-quality columns from the MCMC traces.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from polypesto.core.study import Study


_PERC_COLS = ("percentile_5", "percentile_25", "percentile_75", "percentile_95")


def _param_scale_map(study: Study) -> Dict[str, str]:
    """Return {parameterId: parameterScale} for estimated parameters."""
    ref = next(iter(study.problems.values()))
    pdf = ref.petab_problem.parameter_df
    return pdf["parameterScale"].to_dict()


def _to_fit_scale(value: float, scale: str) -> float:
    """Map a raw-space value into the same scale pypesto reports ensemble stats in."""
    if scale == "log10":
        return float(np.log10(value))
    if scale == "log":
        return float(np.log(value))
    if scale in ("lin", "linear"):
        return float(value)
    raise ValueError(f"Unknown parameterScale: {scale!r}")


def _chain_stats(study: Study) -> pd.DataFrame:
    """burn_in / n_samples / usable_samples / effective sample size per problem."""
    from pypesto.sample import effective_sample_size

    rows = []
    for key, result in (study.results or {}).items():
        row = {"problem_key": str(key), "burn_in": np.nan, "n_samples": np.nan, "ess": np.nan}
        if result is not None and getattr(result, "sample_result", None) is not None:
            s = result.sample_result
            trace = s.get("trace_x", None)
            if trace is not None and hasattr(trace, "shape") and trace.ndim == 3:
                row["n_samples"] = int(trace.shape[1])
            b = s.get("burn_in", None)
            if b is not None:
                row["burn_in"] = int(b)
            try:
                row["ess"] = float(effective_sample_size(result))
            except Exception:
                pass
        rows.append(row)

    df = pd.DataFrame(rows).set_index("problem_key")
    df["usable_samples"] = df["n_samples"] - df["burn_in"]
    df["burn_in_fraction"] = df["burn_in"] / df["n_samples"]
    return df


def recovery_metrics(
    summary: pd.DataFrame,
    study: Optional[Study] = None,
) -> pd.DataFrame:
    """Augment the Study summary with recovery / CI / chain-quality columns.

    Added columns:
      - ``true_value_fit_scale`` : true_value mapped into pypesto's fit scale
      - ``bias``                 : ensemble_median - true_value_fit_scale
      - ``abs_bias``             : |bias|
      - ``ci_width_90``          : percentile_95 - percentile_5   (fit-scale units)
      - ``ci_width_50``          : percentile_75 - percentile_25  (fit-scale units)
      - ``covers_true_90``       : 90% credible interval contains truth
      - ``covers_true_50``       : 50% credible interval contains truth
    If ``study`` is given, also adds ``burn_in``, ``n_samples``, ``usable_samples``,
    ``burn_in_fraction``, and ``ess`` per problem.
    """
    # ``Study.results_summary()`` produces a MultiIndex (problem_key, parameterId)
    # and also leaves a redundant ``parameterId`` column from check_identifiability.
    # Drop the redundant column so reset_index doesn't collide.
    summary = summary.copy()
    if "parameterId" in summary.columns and "parameterId" in (summary.index.names or []):
        summary = summary.drop(columns="parameterId")
    df = summary.reset_index()

    if study is not None:
        scale_map = _param_scale_map(study)
    else:
        scale_map = {pid: "log10" for pid in df["parameterId"].unique()}

    df["parameterScale"] = df["parameterId"].map(scale_map)
    df["true_value_fit_scale"] = [
        _to_fit_scale(v, s) for v, s in zip(df["true_value"], df["parameterScale"])
    ]

    df["bias"] = df["ensemble_median"] - df["true_value_fit_scale"]
    df["abs_bias"] = df["bias"].abs()
    df["ci_width_90"] = df["percentile_95"] - df["percentile_5"]
    df["ci_width_50"] = df["percentile_75"] - df["percentile_25"]
    df["covers_true_90"] = (df["percentile_5"] <= df["true_value_fit_scale"]) & (
        df["true_value_fit_scale"] <= df["percentile_95"]
    )
    df["covers_true_50"] = (df["percentile_25"] <= df["true_value_fit_scale"]) & (
        df["true_value_fit_scale"] <= df["percentile_75"]
    )

    if study is not None:
        chain_df = _chain_stats(study)
        df = df.merge(chain_df, left_on="problem_key", right_index=True, how="left")

    return df.set_index(["problem_key", "parameterId"])


def study_summary(metrics_df: pd.DataFrame) -> pd.Series:
    """Aggregate identifiability/recovery metrics across a whole study."""
    out: Dict[str, float] = {"n_rows": float(len(metrics_df))}

    if "converged" in metrics_df.columns:
        out["frac_converged"] = float(metrics_df["converged"].mean())
        conv = metrics_df[metrics_df["converged"]]
    else:
        out["frac_converged"] = float("nan")
        conv = metrics_df

    if len(conv):
        out["mean_abs_bias"] = float(conv["abs_bias"].mean())
        out["median_ci_width_90"] = float(conv["ci_width_90"].median())
        out["coverage_90"] = float(conv["covers_true_90"].mean())
        out["coverage_50"] = float(conv["covers_true_50"].mean())
    else:
        for k in ("mean_abs_bias", "median_ci_width_90", "coverage_90", "coverage_50"):
            out[k] = float("nan")

    return pd.Series(out)
