from typing import List, Dict

from numpy.typing import ArrayLike
from matplotlib.axes import Axes
import seaborn as sns


def draw_true_param_line(ax: Axes, value: float, orientation: str = "v", **kwargs):
    """Draws a line for the true parameter value on the given axes.

    Args:
        ax (Axes): The axes to draw on.
        value (float): The true parameter value.
        orientation (str, optional): The orientation of the line. Defaults to "v".

    Raises:
        ValueError: _description_
    """

    style = dict(color="red", linestyle="--", linewidth=2)
    style.update(kwargs)

    if orientation == "v":
        ax.axvline(x=value, **style)
    elif orientation == "h":
        ax.axhline(y=value, **style)
    else:
        raise ValueError(f"Unknown orientation: {orientation}. Use 'v' or 'h'.")


def draw_true_param_marker(ax: Axes, x: ArrayLike, y: ArrayLike, **kwargs):
    """Draws a marker for the true parameter values on the given axes.

    Args:
        ax (Axes): The axes to draw on.
        x (ArrayLike): The x-coordinates of the true parameter values.
        y (ArrayLike): The y-coordinates of the true parameter values.
    """

    style = dict(color="red", marker="*", s=150, alpha=0.8, zorder=10)
    style.update(kwargs)
    ax.scatter(x, y, **style)


def plot_true_params_on_pairgrid(grid: sns.PairGrid, true_params: Dict[str, float]):
    """Plots true parameter values on a seaborn PairGrid.

    Args:
        grid (sns.PairGrid): The PairGrid to draw on.
        true_params (Dict[str, float]): The true parameter values.
    """

    # assert param names are in true_params
    param_names = list(true_params.keys())
    if len(param_names) == 0:
        print("No true values to plot.")
        return

    for i, param_y in enumerate(param_names):
        for j, param_x in enumerate(param_names):

            if i >= len(grid.axes) or j >= len(grid.axes[i]):
                continue

            ax = grid.axes[i][j]

            if i == j:
                draw_true_param_line(ax, true_params[param_y], orientation="v")
            else:
                draw_true_param_marker(ax, true_params[param_x], true_params[param_y])


def plot_true_params_on_trace(axes: List[Axes], true_params: Dict[str, float]):
    """Plots true parameter values on a trace plot.

    Args:
        axes (List[Axes]): The axes to draw on.
        true_params (Dict[str, float]): The true parameter values.
    """

    param_names = list(true_params.keys())
    for i, param in enumerate(param_names):
        if i >= len(axes):
            continue

        draw_true_param_line(axes[i], true_params[param], orientation="h")


def plot_true_params_on_distribution(axes: List[Axes], true_params: Dict[str, float]):
    """Plots true parameter values on a distribution plot.

    Args:
        axes (List[Axes]): The axes to draw on.
        true_params (Dict[str, float]): The true parameter values.
    """

    param_names = list(true_params.keys())
    for i, param in enumerate(param_names):
        if i < len(axes) and param in true_params:
            draw_true_param_line(axes[i], true_params[param], orientation="v")
