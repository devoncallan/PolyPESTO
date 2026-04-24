"""Visualization helpers for identifiability studies.

Lives outside ``polypesto/`` alongside ``metrics.py``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from polypesto.core.study import Study


_LOG10_TICKS = [-3, -2, -1, 0, 1, 2]
_LOG10_LABELS = ["0.001", "0.01", "0.1", "1", "10", "100"]
_LOG10_LIM = (-3, 2)


def _hpd_threshold(kde: gaussian_kde, samples: np.ndarray, level: float) -> float:
    """Density level above which a fraction ``level`` of posterior mass lies.

    Standard KDE-based HPD: evaluate the density at each sample and take the
    ``(1 - level)``-quantile. Points with density above this threshold form
    the ``level``-HPD region.
    """
    d = kde(samples)
    return float(np.quantile(d, 1.0 - level))


def plot_rA_rB_posteriors(
    study: Study,
    level: float = 0.95,
    show_nonconverged: bool = True,
    ax: Optional[Axes] = None,
    grid_n: int = 120,
) -> Tuple[Figure, Axes]:
    """Overlay 2D posteriors in (log10 rA, log10 rB) space.

    For each problem: draw the KDE contour at the ``level`` HPD (default 95%),
    a dot at the posterior median, a star at the true (rA, rB). Non-converged
    problems are drawn only as an X at the truth when ``show_nonconverged``.

    Axes are in log10 units; tick labels show the corresponding linear values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
    else:
        fig = ax.figure

    # Color per true-parameter set so rows with the same truth share a color.
    param_ids = sorted({key.param_id for key in study.problems.keys()})
    cmap = plt.get_cmap("tab10")
    color_map = {pid: cmap(i % 10) for i, pid in enumerate(param_ids)}

    xx, yy = np.meshgrid(
        np.linspace(_LOG10_LIM[0], _LOG10_LIM[1], grid_n),
        np.linspace(_LOG10_LIM[0], _LOG10_LIM[1], grid_n),
    )

    for key, prob in study.problems.items():
        ens = prob.get_ensemble()
        param_set = study.get_true_params(key.param_id)
        true_log_rA = float(np.log10(param_set["rA"]))
        true_log_rB = float(np.log10(param_set["rB"]))
        color = color_map[key.param_id]

        if ens is None or ens.x_vectors.size == 0:
            if show_nonconverged:
                ax.plot(true_log_rA, true_log_rB, marker="x", color=color,
                        linestyle="none", markersize=14, mew=2.5)
            continue

        x_names = list(ens.x_names)
        samples = np.stack(
            [ens.x_vectors[x_names.index("rA")], ens.x_vectors[x_names.index("rB")]]
        )
        kde = gaussian_kde(samples)
        threshold = _hpd_threshold(kde, samples, level)
        zz = kde(np.stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[threshold], colors=[color], linewidths=1.8)
        ax.plot(np.median(samples[0]), np.median(samples[1]),
                marker="o", color=color, linestyle="none", markersize=5)
        ax.plot(true_log_rA, true_log_rB,
                marker="*", color=color, linestyle="none",
                markersize=16, mec="k", mew=0.6)

    # rA = rB reference
    ax.plot(_LOG10_LIM, _LOG10_LIM, color="grey", linestyle=":", linewidth=1, zorder=0)
    ax.set_xlim(*_LOG10_LIM)
    ax.set_ylim(*_LOG10_LIM)

    ax.set_xticks(_LOG10_TICKS)
    ax.set_xticklabels(_LOG10_LABELS)
    ax.set_yticks(_LOG10_TICKS)
    ax.set_yticklabels(_LOG10_LABELS)
    ax.set_xlabel(r"$r_A$")
    ax.set_ylabel(r"$r_B$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend: param-set colors + glyph key
    truth_handles = [
        Line2D([0], [0], marker="*", color=c, linestyle="none", markersize=12,
               mec="k", mew=0.5, label=pid)
        for pid, c in color_map.items()
    ]
    glyph_handles = [
        Line2D([0], [0], marker="*", color="0.3", linestyle="none", markersize=12,
               mec="k", mew=0.5, label="truth"),
        Line2D([0], [0], marker="o", color="0.3", linestyle="none", markersize=6,
               label="posterior median"),
        Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.8,
               label=f"{int(level * 100)}% HPD"),
    ]
    if show_nonconverged:
        glyph_handles.append(
            Line2D([0], [0], marker="x", color="0.3", linestyle="none",
                   markersize=10, mew=2, label="non-converged")
        )
    leg1 = ax.legend(handles=truth_handles, title="true params",
                     loc="upper left", fontsize=9, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=glyph_handles, loc="lower right", fontsize=9)

    return fig, ax


def plot_composition_vs_feed(
    study: Study, ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Scatter simulated FA (with noise error bars) against feed fraction fA.

    One series per true-parameter set. Assumes the simulated measurements live
    in each problem's PEtab ``measurement_df`` keyed by conditionId, with A0/B0
    in the ``condition_df``. fA is computed as A0/(A0+B0).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
    else:
        fig = ax.figure

    param_ids = sorted({key.param_id for key in study.problems.keys()})
    cmap = plt.get_cmap("tab10")
    color_map = {pid: cmap(i % 10) for i, pid in enumerate(param_ids)}
    seen: set[str] = set()

    for key, prob in study.problems.items():
        meas = prob.petab_problem.measurement_df
        cond = prob.petab_problem.condition_df
        df = meas.merge(
            cond[["A0", "B0"]], left_on="simulationConditionId", right_index=True
        )
        fA = df["A0"] / (df["A0"] + df["B0"])
        FA = df["measurement"]
        sigma = df["noiseParameters"].astype(float)
        label = key.param_id if key.param_id not in seen else None
        seen.add(key.param_id)
        ax.errorbar(fA, FA, yerr=sigma, fmt="o",
                    color=color_map[key.param_id], markersize=5,
                    capsize=3, label=label)

    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.4, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"feed fraction $f_A$")
    ax.set_ylabel(r"copolymer composition $F_A$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(title="true params", loc="upper left", fontsize=9, title_fontsize=9)
    return fig, ax
