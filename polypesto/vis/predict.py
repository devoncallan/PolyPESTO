from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


import pypesto.visualize as vis
from pypesto.ensemble import EnsemblePrediction

from polypesto.core.problem import Problem
from .base import safe_plot

########################
### Prediction Plots ###
########################


# @safe_plot
def plot_ensemble_predictions(
    ensemble_pred: EnsemblePrediction, prob: Problem, levels=[90, 95, 99], **kwargs
) -> Tuple[Figure, Axes]:

    mdf = prob.petab_problem.measurement_df
    mdf["conditionId"] = mdf["simulationConditionId"]

    axs = vis.sampling_prediction_trajectories(
        ensemble_prediction=ensemble_pred,
        levels=levels,
        measurement_df=mdf,
        groupby="condition",
        **kwargs,
    )
    # axs = sampling_prediction_trajectories(
    #     ensemble_prediction=ensemble_pred, levels=levels, measurement_df=mdf, **kwargs
    # )

    fig = plt.gcf()
    plt.tight_layout()

    return fig, axs


import logging
import warnings
from collections.abc import Sequence
from colorsys import rgb_to_hls
from typing import Optional, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from pypesto.C import (
    CONDITION,
    LEN_RGB,
    MEDIAN,
    OUTPUT,
    RGB,
    RGBA_BLACK,
    RGBA_MAX,
    RGBA_MIN,
    STANDARD_DEVIATION,
)
from pypesto.ensemble import EnsemblePrediction, get_percentile_label
from pypesto.result import McmcPtResult, PredictionResult, Result
from pypesto.sample import calculate_ci_mcmc_sample
from pypesto.visualize.misc import rgba2rgb

cmap = matplotlib.cm.viridis
logger = logging.getLogger(__name__)


prediction_errorbar_settings = {
    "fmt": "none",
    "color": "k",
    "capsize": 10,
}


def _get_level_percentiles(level: float) -> tuple[float, float]:
    """Convert a credibility level to percentiles.

    Similar to the highest-density region of a symmetric, unimodal distribution
    (e.g. Gaussian distribution).

    For example, an credibility level of `95` will be converted to
    `(2.5, 97.5)`.

    Parameters
    ----------
    level:
        The credibility level used to calculate the percentiles. For example,
        `[95]` for a 95% credibility interval. These levels are split
        symmetrically, e.g. `95` corresponds to plotting values between the
        2.5% and 97.5% percentiles, and are equivalent to highest-density
        regions for a normal distribution. For skewed distributions, asymmetric
        percentiles may be preferable, but are not yet implemented.

    Returns
    -------
    The percentiles, with the lower percentile first.
    """
    lower_percentile = (100 - level) / 2
    return lower_percentile, 100 - lower_percentile


def _get_statistic_data(
    summary: dict[str, PredictionResult],
    statistic: str,
    condition_id: str,
    output_id: str,
) -> tuple[Sequence[float], Sequence[float]]:
    """Get statistic-, condition-, and output-specific data.

    Parameters
    ----------
    summary:
        A `pypesto.ensemble.EnsemblePrediction.prediction_summary`, used as the
        source of annotated data to subset.
    statistic:
        Select data for a specific statistic by its label, e.g. `MEDIAN` or
        `get_percentile_label(95)`.
    condition_id:
        Select data for a specific condition by its ID.
    output_id:
        Select data for a specific output by its ID.

    Returns
    -------
    Predicted values and their corresponding time points. A tuple of two
    sequences, where the first sequence is time points, and the second
    sequence is predicted values at the corresponding time points.
    """
    condition_index = summary[statistic].condition_ids.index(condition_id)
    condition_result = summary[statistic].conditions[condition_index]
    t = condition_result.timepoints
    output_index = condition_result.output_ids.index(output_id)
    y = condition_result.output[:, output_index]
    return (t, y)


def _plot_trajectories_by_condition(
    summary: dict[str, PredictionResult],
    condition_ids: Sequence[str],
    output_ids: Sequence[str],
    axes: matplotlib.axes.Axes,
    levels: Sequence[float],
    level_opacities: dict[int, float],
    labels: dict[str, str],
    variable_colors: Sequence[RGB],
    average: str = MEDIAN,
    add_sd: bool = False,
    grouped_measurements: dict[tuple[str, str], Sequence[Sequence[float]]] = None,
) -> None:
    """Plot predicted trajectories, with subplots grouped by condition.

    Parameters
    ----------
    summary:
        A `pypesto.ensemble.EnsemblePrediction.prediction_summary`, used as the
        source of annotated data to plot.
    condition_ids:
        The IDs of conditions to plot.
    output_ids:
        The IDs of outputs to plot.
    axes:
        The axes to plot with. Should contain atleast `len(output_ids)`
        subplots.
    levels:
        Credibility levels, e.g. [95] for a 95% credibility interval. See the
        :py:func:`_get_level_percentiles` method for a description of how these
        levels are handled, and current limitations.
    level_opacities:
        A mapping from the credibility levels to the opacities that they should
        be plotted with. Opacity is the only thing that differentiates
        credibility levels in the resulting plot.
    labels:
        Keys should be ensemble output IDs, values should be the desired
        label for that output. Defaults to output IDs.
    variable_colors:
        Colors used to differentiate plotted outputs. The order should
        correspond to `output_ids`.
    average:
        The ID of the statistic that will be plotted as the average (e.g.,
        `MEDIAN` or `MEAN`).
    add_sd:
        Whether to add the standard deviation of the predictions to the plot.
    grouped_measurements:
        Measurement data that has already been grouped by condition and output,
        where the keys are `(condition_id, output_id)` 2-tuples, and the values
        are `[sequence of x-axis values, sequence of y-axis values]`.
    """
    # Each subplot has all data for a single condition.
    for condition_index, condition_id in enumerate(condition_ids):
        ax = axes.flat[condition_index]
        ax.set_title(f"Condition: {labels[condition_id]}")
        # Each subplot has all data for all condition-specific outputs.
        for output_index, output_id in enumerate(output_ids):
            facecolor0 = variable_colors[output_index]
            # Plot the average for each output.
            t_average, y_average = _get_statistic_data(
                summary,
                average,
                condition_id,
                output_id,
            )
            # ax.plot(
            #     t_average,
            #     y_average,
            #     "-",
            #     color=facecolor0,
            #     alpha=0.5
            # )
            if add_sd:
                t_std, y_std = _get_statistic_data(
                    summary,
                    STANDARD_DEVIATION,
                    condition_id,
                    output_id,
                )
                if (t_std != t_average).all():
                    raise ValueError(
                        "Unknown error: timepoints for average and standard "
                        "deviation do not match."
                    )
                ax.errorbar(
                    t_average,
                    y_average,
                    yerr=y_std,
                    **prediction_errorbar_settings,
                )
            # Plot the regions described by the credibility level,
            # for each output.
            for level_index, level in enumerate(levels):
                # Get the percentiles that correspond to the credibility level,
                # as their labels in the `summary`.
                lower_label, upper_label = (
                    get_percentile_label(percentile)
                    for percentile in _get_level_percentiles(level)
                )
                # Get the data for each percentile.
                t_lower, lower_data = _get_statistic_data(
                    summary,
                    lower_label,
                    condition_id,
                    output_id,
                )
                t_upper, upper_data = _get_statistic_data(
                    summary,
                    upper_label,
                    condition_id,
                    output_id,
                )
                # Timepoints must match, or `upper_data` will be plotted at
                # some incorrect time points.
                if not (np.array(t_lower) == np.array(t_upper)).all():
                    raise ValueError(
                        "The timepoints of the data for the upper and lower "
                        "percentiles do not match."
                    )
                # Plot a shaded region between the data that correspond to the
                # lower and upper percentiles.
                ax.fill_between(
                    t_lower,
                    lower_data,
                    upper_data,
                    facecolor=rgba2rgb(
                        variable_colors[output_index] + [level_opacities[level_index]]
                    ),
                    lw=0,
                )
            if measurements := grouped_measurements.get(
                (condition_id, output_id), False
            ):
                ax.scatter(
                    measurements[0],
                    measurements[1],
                    marker="o",
                    facecolor=facecolor0,
                    edgecolor=(
                        "white" if rgb_to_hls(*facecolor0)[1] < 0.5 else "black"
                    ),
                )


def _plot_trajectories_by_output(
    summary: dict[str, PredictionResult],
    condition_ids: Sequence[str],
    output_ids: Sequence[str],
    axes: matplotlib.axes.Axes,
    levels: Sequence[float],
    level_opacities: dict[int, float],
    labels: dict[str, str],
    variable_colors: Sequence[RGB],
    average: str = MEDIAN,
    add_sd: bool = False,
    grouped_measurements: dict[tuple[str, str], Sequence[Sequence[float]]] = None,
) -> None:
    """Plot predicted trajectories, with subplots grouped by output.

    Each subplot is further divided by conditions, such that all conditions
    are displayed side-by-side for a single output. Hence, in each subplot, the
    timepoints of each condition plot are shifted by the the end timepoint of
    the previous condition plot. For examples of this, see the plots with
    `groupby=OUTPUT` in the example notebook
    `doc/example/sampling_diagnostics.ipynb`.

    See :py:func:`_plot_trajectories_by_condition` for parameter descriptions.
    """
    # Each subplot has all data for a single output.
    for output_index, output_id in enumerate(output_ids):
        # Store the end timepoint of the previous condition plot, such that the
        # next condition plot starts at the end of the previous condition plot.
        t0 = 0
        ax = axes.flat[output_index]
        ax.set_title(f"Trajectory: {labels[output_id]}")
        # Each subplot is divided by conditions, with vertical lines.
        for condition_index, condition_id in enumerate(condition_ids):
            facecolor0 = variable_colors[condition_index]
            if condition_index != 0:
                ax.axvline(
                    t0,
                    linewidth=2,
                    color="k",
                )

            t_max = t0
            t_average, y_average = _get_statistic_data(
                summary,
                average,
                condition_id,
                output_id,
            )
            # Shift the timepoints for the average plot to start at the end of
            # the previous condition plot.
            t_average_shifted = t_average + t0
            ax.plot(
                t_average_shifted,
                y_average,
                "k-",
            )
            if add_sd:
                t_std, y_std = _get_statistic_data(
                    summary,
                    STANDARD_DEVIATION,
                    condition_id,
                    output_id,
                )
                if (t_std != t_average).all():
                    raise ValueError(
                        "Unknown error: timepoints for average and standard "
                        "deviation do not match."
                    )
                ax.errorbar(
                    t_average_shifted,
                    y_average,
                    yerr=y_std,
                    **prediction_errorbar_settings,
                )
            t_max = max(t_max, *t_average_shifted)
            for level_index, level in enumerate(levels):
                # Get the percentiles that correspond to the credibility level,
                # as their labels in the `summary`.
                lower_label, upper_label = (
                    get_percentile_label(percentile)
                    for percentile in _get_level_percentiles(level)
                )
                # Get the data for each percentile.
                t_lower, lower_data = _get_statistic_data(
                    summary,
                    lower_label,
                    condition_id,
                    output_id,
                )
                t_upper, upper_data = _get_statistic_data(
                    summary,
                    upper_label,
                    condition_id,
                    output_id,
                )
                # Shift the timepoints for the `fill_between` plots to start at
                # the end of the previous condition plot.
                t_lower_shifted = t_lower + t0
                t_upper_shifted = t_upper + t0
                # Timepoints must match, or `upper_data` will be plotted at
                # some incorrect time points.
                if not (np.array(t_lower) == np.array(t_upper)).all():
                    raise ValueError(
                        "The timepoints of the data for the upper and lower "
                        "percentiles do not match."
                    )
                # Plot a shaded region between the data that correspond to the
                # lower and upper percentiles.
                ax.fill_between(
                    t_lower_shifted,
                    lower_data,
                    upper_data,
                    facecolor=rgba2rgb(facecolor0 + [level_opacities[level_index]]),
                    lw=0,
                )
                t_max = max(t_max, *t_lower_shifted, *t_upper_shifted)
            if measurements := grouped_measurements.get(
                (condition_id, output_id), False
            ):
                ax.scatter(
                    [t0 + _t for _t in measurements[0]],
                    measurements[1],
                    marker="o",
                    facecolor=facecolor0,
                    edgecolor=(
                        "white" if rgb_to_hls(*facecolor0)[1] < 0.5 else "black"
                    ),
                )
            # Set t0 to the last plotted timepoint of the current condition
            # plot.
            t0 = t_max


def _get_condition_and_output_ids(
    summary: dict[str, PredictionResult],
) -> tuple[Sequence[str], Sequence[str]]:
    """Get all condition and output IDs in a prediction summary.

    Parameters
    ----------
    summary:
        The prediction summary to extract condition and output IDs from.

    Returns
    -------
    A 2-tuple, with the following indices and values.
    - `0`: a list of all condition IDs.
    - `1`: a list of all output IDs.
    """
    # For now, all prediction results must predict for the same set of
    # conditions. Can support different conditions later.
    all_condition_ids = [prediction.condition_ids for prediction in summary.values()]
    if not (
        np.array(
            [
                set(condition_ids) == set(all_condition_ids[0])
                for condition_ids in all_condition_ids
            ]
        ).all()
    ):
        raise KeyError("All predictions must have the same set of conditions.")
    condition_ids = all_condition_ids[0]

    output_ids = sorted(
        {
            output_id
            for prediction in summary.values()
            for condition in prediction.conditions
            for output_id in condition.output_ids
        }
    )

    return condition_ids, output_ids


def _handle_legends(
    fig: matplotlib.figure.Figure,
    axes: matplotlib.axes.Axes,
    levels: Union[float, Sequence[float]],
    labels: dict[str, str],
    level_opacities: Sequence[float],
    variable_names: Sequence[str],
    variable_colors: Sequence[RGB],
    groupby: str,
    artist_padding: float,
    n_col: int,
    average: str,
    add_sd: bool,
    grouped_measurements: Optional[dict[tuple[str, str], Sequence[Sequence[float]]]],
) -> None:
    """Add legends to a sampling prediction trajectories plot.

    Create a dummy plot from fake data such that it can be used to produce
    appropriate legends.

    Variable here refers to the thing that differs in the plot. For example, if
    the call to :py:func:`sampling_prediction_trajectories` has
    `groupby=OUTPUT`, then the variable is `CONDITION`. Similarly, if
    `groupby=CONDITION`, then the variable is `OUTPUT`.

    Parameters
    ----------
    fig:
        The figure to add the legends to.
    axes:
        The axes of the figure to add the legend to.
    levels:
        The credibility levels.
    labels:
        The labels for the IDs in the plot.
    level_opacities:
        The opacity to plot each credibility level with.
    variable_names:
        The name of each variable.
    variable_colors:
        The color to plot each variable in.
    groupby:
        The grouping of data in the subplots.
    artist_padding:
        The padding between the figure and the legends.
    n_col:
        The number of columns of subplots in the figure.
    average:
        The ID of the statistic that will be plotted as the average (e.g.,
        `MEDIAN` or `MEAN`).
    add_sd:
        Whether to add the standard deviation of the predictions to the plot.
    grouped_measurements:
        Measurement data that has already been grouped by condition and output,
        where the keys are `(condition_id, output_id)` 2-tuples, and the values
        are `[sequence of x-axis values, sequence of y-axis values]`.
    """
    # Fake plots for legend line styles
    fake_data = [[0], [0]]
    variable_lines = np.array(
        [
            # Assumes that the color for a variable is always the same, with
            # different opacity for different credibility interval levels.
            # Create a line object with fake data for each variable value.
            [
                labels[variable_name],
                Line2D(*fake_data, color=variable_colors[index], lw=4),
            ]
            for index, variable_name in enumerate(variable_names)
        ]
    )
    # Assumes that different CI levels are represented as
    # different opacities of the same color.
    # Create a line object with fake data for each credibility level.
    ci_lines = []
    for index, level in enumerate(levels):
        ci_lines.append(
            [
                f"{level}% CI",
                Line2D(
                    *fake_data,
                    color=rgba2rgb([*RGBA_BLACK[:LEN_RGB], level_opacities[index]]),
                    lw=4,
                ),
            ]
        )

    # Create a line object with fake data for the average line.
    average_title = average.title()
    average_line_object_line2d = Line2D(*fake_data, color=RGBA_BLACK)
    if add_sd:
        capline = Line2D(
            *fake_data,
            color=prediction_errorbar_settings["color"],
            # https://github.com/matplotlib/matplotlib/blob
            # /710fce3df95e22701bd68bf6af2c8adbc9d67a79/lib/matplotlib/
            # axes/_axes.py#L3424=
            markersize=2.0 * prediction_errorbar_settings["capsize"],
        )
        average_title += " + SD"
        barline = LineCollection(
            np.empty((2, 2, 2)),
            color=prediction_errorbar_settings["color"],
        )
        average_line_object = ErrorbarContainer(
            (
                average_line_object_line2d,
                [capline],
                [barline],
            ),
            has_yerr=True,
        )
    else:
        average_line_object = average_line_object_line2d
    average_line = [[average_title, average_line_object]]

    # Create a line object with fake data for the data points.
    data_line = []
    if grouped_measurements:
        data_line = [
            [
                "Data",
                Line2D(
                    *fake_data,
                    linewidth=0,
                    marker="o",
                    markerfacecolor="grey",
                    markeredgecolor="white",
                ),
            ]
        ]

    level_lines = np.array(ci_lines + average_line + data_line)

    # CI level, and variable name, legends.
    legend_options_top_right = {
        "bbox_to_anchor": (1 + artist_padding, 1),
        "loc": "upper left",
    }
    legend_options_bottom_right = {
        "bbox_to_anchor": (1 + artist_padding, 0),
        "loc": "lower left",
    }
    legend_titles = {
        OUTPUT: "Conditions",
        CONDITION: "Trajectories",
    }
    legend_variables = axes.flat[n_col - 1].legend(
        variable_lines[:, 1],
        variable_lines[:, 0],
        **legend_options_top_right,
        title=legend_titles[groupby],
    )
    # Legend for CI levels
    axes.flat[-1].legend(
        level_lines[:, 1],
        level_lines[:, 0],
        **legend_options_bottom_right,
        title="Prediction",
    )
    fig.add_artist(legend_variables)


def _handle_colors(
    levels: Union[float, Sequence[float]],
    n_variables: int,
    reverse: bool = False,
) -> tuple[Sequence[float], Sequence[RGB]]:
    """Calculate the colors for the prediction trajectories plot.

    Parameters
    ----------
    levels:
        The credibility levels.
    n_variables:
        The maximum possible number of variables per subplot.

    Returns
    -------
    A 2-tuple, with the following indices and values.
    - `0`: a list of opacities, one per level.
    - `1`: a list of colors, one per variable.
    """
    level_opacities = sorted(
        # min 30%, max 100%, opacity
        np.linspace(0.3 * RGBA_MAX, RGBA_MAX, len(levels)),
        reverse=reverse,
    )
    cmap_min = RGBA_MIN
    cmap_max = 0.85 * (RGBA_MAX - RGBA_MIN) + RGBA_MIN  # exclude yellows

    # define colormap
    variable_colors = [
        list(cmap(v))[:LEN_RGB] for v in np.linspace(cmap_min, cmap_max, n_variables)
    ]

    return level_opacities, variable_colors


def sampling_prediction_trajectories(
    ensemble_prediction: EnsemblePrediction,
    levels: Union[float, Sequence[float]],
    title: str = None,
    size: tuple[float, float] = None,
    axes: matplotlib.axes.Axes = None,
    labels: dict[str, str] = None,
    axis_label_padding: int = 50,
    groupby: str = CONDITION,
    condition_gap: float = 0.01,
    condition_ids: Sequence[str] = None,
    output_ids: Sequence[str] = None,
    weighting: bool = False,
    reverse_opacities: bool = False,
    average: str = MEDIAN,
    add_sd: bool = False,
    measurement_df: pd.DataFrame = None,
) -> matplotlib.axes.Axes:
    """
    Visualize prediction trajectory of an EnsemblePrediction.

    Plot MCMC-based prediction credibility intervals for the
    model states or outputs. One or various credibility levels
    can be depicted. Plots are grouped by condition.

    Parameters
    ----------
    ensemble_prediction:
        The ensemble prediction.
    levels:
        Credibility levels, e.g. [95] for a 95% credibility interval. See the
        :py:func:`_get_level_percentiles` method for a description of how these
        levels are handled, and current limitations.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    axes:
        Axes object to use.
    labels:
        Keys should be ensemble output IDs, values should be the desired
        label for that output. Defaults to output IDs.
    axis_label_padding:
        Pixels between axis labels and plots.
    groupby:
        Group plots by `pypesto.C.OUTPUT` or
        `pypesto.C.CONDITION`.
    condition_gap:
        Gap between conditions when
        `groupby == pypesto.C.CONDITION`.
    condition_ids:
        If provided, only data for the provided condition IDs will be plotted.
    output_ids:
        If provided, only data for the provided output IDs will be plotted.
    weighting:
        Whether weights should be used for trajectory.
    reverse_opacities:
        Whether to reverse the opacities that are assigned to different levels.
    average:
        The ID of the statistic that will be plotted as the average (e.g.,
        `MEDIAN` or `MEAN`).
    add_sd:
        Whether to add the standard deviation of the predictions to the plot.
    measurement_df:
        Plot measurement data. NB: This should take the form of a PEtab
        measurements table, and the `observableId` column should correspond
        to the output IDs in the ensemble prediction.

    Returns
    -------
    axes:
        The plot axes.
    """
    if labels is None:
        labels = {}
    if len(list(levels)) == 1:
        levels = list(levels)
    levels = sorted(levels, reverse=True)
    # Get the percentiles that correspond to the requested credibility levels.
    percentiles = [
        percentile for level in levels for percentile in _get_level_percentiles(level)
    ]

    summary = ensemble_prediction.compute_summary(
        percentiles_list=percentiles, weighting=weighting
    )

    all_condition_ids, all_output_ids = _get_condition_and_output_ids(summary)
    if condition_ids is None:
        condition_ids = all_condition_ids
    condition_ids = list(condition_ids)
    if output_ids is None:
        output_ids = all_output_ids
    output_ids = list(output_ids)

    # Handle data
    grouped_measurements = {}
    if measurement_df is not None:
        import petab.v1 as petab

        for condition_id in condition_ids:
            if petab.PARAMETER_SEPARATOR in condition_id:
                (
                    preequilibration_condition_id,
                    simulation_condition_id,
                ) = condition_id.split(petab.PARAMETER_SEPARATOR)
            else:
                preequilibration_condition_id, simulation_condition_id = (
                    "",
                    condition_id,
                )
            condition = {
                petab.SIMULATION_CONDITION_ID: simulation_condition_id,
            }
            if preequilibration_condition_id:
                condition[petab.PREEQUILIBRATION_CONDITION_ID] = (
                    preequilibration_condition_id
                )
            for output_id in output_ids:
                _df = petab.get_rows_for_condition(
                    measurement_df=measurement_df,
                    condition=condition,
                )
                _df = _df.loc[_df[petab.OBSERVABLE_ID] == output_id]
                grouped_measurements[(condition_id, output_id)] = [
                    _df[petab.TIME],
                    _df[petab.MEASUREMENT],
                ]
        print(grouped_measurements)

    # Set default labels for any unspecified labels.
    labels = {id_: labels.get(id_, id_) for id_ in condition_ids + output_ids}

    if groupby == CONDITION:
        n_variables = len(output_ids)
        variable_names = output_ids
        n_subplots = len(condition_ids)
    elif groupby == OUTPUT:
        n_variables = len(condition_ids)
        variable_names = condition_ids
        n_subplots = len(output_ids)
    else:
        raise ValueError(f"Unsupported groupby value: {groupby}")

    level_opacities, variable_colors = _handle_colors(
        levels=levels,
        n_variables=n_variables,
        reverse=reverse_opacities,
    )
    # variable_colors = [[98/255., 169/255., 143/255.], [45/255., 105/255., 178/255.]]

    if axes is None:
        n_row = int(np.round(np.sqrt(n_subplots)))
        n_col = int(np.ceil(n_subplots / n_row))
        fig, axes = plt.subplots(n_row, n_col, figsize=size, squeeze=False)
        for ax in axes.flat[n_subplots:]:
            ax.remove()
    else:
        fig = axes.get_figure()
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        if len(axes.flat) < n_subplots:
            raise ValueError(
                "Provided `axes` contains insufficient subplots. At least "
                f"{n_subplots} are required."
            )
    artist_padding = axis_label_padding / (fig.get_size_inches() * fig.dpi)[0]

    if groupby == CONDITION:
        _plot_trajectories_by_condition(
            summary=summary,
            condition_ids=condition_ids,
            output_ids=output_ids,
            axes=axes,
            levels=levels,
            level_opacities=level_opacities,
            labels=labels,
            variable_colors=variable_colors,
            average=average,
            add_sd=add_sd,
            grouped_measurements=grouped_measurements,
        )
    elif groupby == OUTPUT:
        _plot_trajectories_by_output(
            summary=summary,
            condition_ids=condition_ids,
            output_ids=output_ids,
            axes=axes,
            levels=levels,
            level_opacities=level_opacities,
            labels=labels,
            variable_colors=variable_colors,
            average=average,
            add_sd=add_sd,
            grouped_measurements=grouped_measurements,
        )

    # if title:
    #     fig.suptitle(title)

    # _handle_legends(
    #     fig=fig,
    #     axes=axes,
    #     levels=levels,
    #     labels=labels,
    #     level_opacities=level_opacities,
    #     variable_names=variable_names,
    #     variable_colors=variable_colors,
    #     groupby=groupby,
    #     artist_padding=artist_padding,
    #     n_col=n_col,
    #     average=average,
    #     add_sd=add_sd,
    #     grouped_measurements=grouped_measurements,
    # )

    # X and Y labels
    # xmin = min(ax.get_position().xmin for ax in axes.flat)
    # ymin = min(ax.get_position().ymin for ax in axes.flat)
    # xlabel = "Cumulative time across all conditions" if groupby == OUTPUT else "Time"
    # fig.text(
    #     0.5,
    #     ymin - artist_padding,
    #     xlabel,
    #     ha="center",
    #     va="center",
    #     transform=fig.transFigure,
    # )
    # fig.text(
    #     xmin - artist_padding,
    #     0.5,
    #     "Simulated values",
    #     ha="center",
    #     va="center",
    #     transform=fig.transFigure,
    #     rotation="vertical",
    # )

    # plt.tight_layout()  # Ruins layout for `groupby == OUTPUT`.
    return axes
