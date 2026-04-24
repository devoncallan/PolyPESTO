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


_LOG10_TICKS = [-2, -1, 0, 1, 2]
_LOG10_LABELS = ["0.01", "0.1", "1", "10", "100"]


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

    if study.ensembles is None or study.results is None:
        raise ValueError("Study must have results/ensembles populated.")

    # Color per true-parameter set so rows with the same truth share a color.
    param_ids = sorted({key.param_id for key in study.problems.keys()})
    cmap = plt.get_cmap("tab10")
    color_map = {pid: cmap(i % 10) for i, pid in enumerate(param_ids)}

    xs_all: list[float] = []
    ys_all: list[float] = []

    for key, ens in study.ensembles.items():
        param_set = study.get_true_params(key.param_id)
        true_log_rA = float(np.log10(param_set["rA"]))
        true_log_rB = float(np.log10(param_set["rB"]))
        color = color_map[key.param_id]
        xs_all.extend([true_log_rA])
        ys_all.extend([true_log_rB])

        converged = ens is not None and ens.x_vectors.size > 0

        if not converged:
            if show_nonconverged:
                ax.plot(
                    true_log_rA, true_log_rB,
                    marker="x", markersize=14, mew=2.5, color=color,
                    linestyle="none",
                )
            continue

        # Pull (rA, rB) samples; samples are in log10 (fit) scale already.
        x_names = list(ens.x_names)
        i_rA = x_names.index("rA")
        i_rB = x_names.index("rB")
        samples = np.stack([ens.x_vectors[i_rA], ens.x_vectors[i_rB]], axis=0)
        xs_all.extend(samples[0].tolist())
        ys_all.extend(samples[1].tolist())

        kde = gaussian_kde(samples)
        threshold = _hpd_threshold(kde, samples, level)

        lo_x = min(samples[0].min(), true_log_rA) - 0.3
        hi_x = max(samples[0].max(), true_log_rA) + 0.3
        lo_y = min(samples[1].min(), true_log_rB) - 0.3
        hi_y = max(samples[1].max(), true_log_rB) + 0.3
        xx, yy = np.meshgrid(
            np.linspace(lo_x, hi_x, grid_n),
            np.linspace(lo_y, hi_y, grid_n),
        )
        zz = kde(np.stack([xx.ravel(), yy.ravel()], axis=0)).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[threshold], colors=[color], linewidths=1.8)

        ax.plot(
            np.median(samples[0]), np.median(samples[1]),
            marker="o", markersize=5, color=color, linestyle="none",
        )
        ax.plot(
            true_log_rA, true_log_rB,
            marker="*", markersize=16, mec="k", mew=0.6, color=color,
            linestyle="none",
        )

    # rA = rB reference
    x_lo, x_hi = min(xs_all) - 0.3, max(xs_all) + 0.3
    y_lo, y_hi = min(ys_all) - 0.3, max(ys_all) + 0.3
    lo = min(x_lo, y_lo, -2.2)
    hi = max(x_hi, y_hi, 2.2)
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle=":", linewidth=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

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
