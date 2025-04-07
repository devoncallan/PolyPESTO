from typing import List, Dict

from numpy.typing import ArrayLike
from matplotlib.axes import Axes
import seaborn as sns


def draw_true_param_line(ax: Axes, value: float, orientation: str = "v", **kwargs):

    style = dict(color="red", linestyle="--", linewidth=2)
    style.update(kwargs)

    if orientation == "v":
        ax.axvline(x=value, **style)
    elif orientation == "h":
        ax.axhline(y=value, **style)
    else:
        raise ValueError(f"Unknown orientation: {orientation}. Use 'v' or 'h'.")


def draw_true_param_marker(ax: Axes, x: ArrayLike, y: ArrayLike, **kwargs):

    style = dict(color="red", marker="*", s=150, alpha=0.8, zorder=10)
    style.update(kwargs)
    ax.scatter(x, y, **style)


def plot_true_params_on_pairgrid(
    grid: sns.PairGrid, true_params: dict, param_names: List[str]
):

    # assert param names are in true_params
    true_values = {
        param: true_params[param] for param in param_names if param in true_params
    }
    if len(true_values) == 0:
        print("No true values to plot.")
        return

    for i, param_y in enumerate(param_names):
        for j, param_x in enumerate(param_names):

            if i >= len(grid.axes) or j >= len(grid.axes[i]):
                continue

            ax = grid.axes[i][j]

            if i == j:
                draw_true_param_line(ax, true_values[param_y], orientation="v")
            else:
                draw_true_param_marker(ax, true_values[param_x], true_values[param_y])


def plot_true_params_on_trace(
    axes: List[Axes], true_params: dict, param_names: List[str]
):

    # assert param names are in true_params
    true_values = {
        param: true_params[param] for param in param_names if param in true_params
    }
    if len(true_values) == 0:
        print("No true values to plot.")
        return

    for i, param in enumerate(param_names):
        if i >= len(axes):
            continue

        draw_true_param_line(axes[i], true_values[param], orientation="h")


def plot_true_params_on_distribution(
    axes: List[Axes], true_params: Dict[str, float], param_names: List[str]
):

    for i, param in enumerate(param_names):
        if i < len(axes) and param in true_params:
            draw_true_param_line(axes[i], true_params[param], orientation="v")
