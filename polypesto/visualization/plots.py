"""
Visualization functions for parameter estimation results.

This module provides plotting functions focused exclusively on visualization of
parameter estimation results from optimization, profiling, and sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

# Import our result handlers
from polypesto.core.results import (
    ParameterResult,
    OptimizationResult,
    ProfileResult,
    SamplingResult,
)


def plot_empty_figure(msg="No data available", figsize=(10, 6)):
    """
    Create an empty figure with a message.

    Parameters
    ----------
    msg : str
        Message to display
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(
        0.5,
        0.5,
        msg,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.tight_layout()
    return fig, ax


def plot_waterfall(result, **kwargs):
    """
    Create a waterfall plot from optimization results.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing optimization results
    **kwargs
        Additional arguments passed to pypesto.visualize.waterfall
        figsize: tuple - Figure size
        scale_y: str - Scale for y-axis ('log10' or 'lin')

    Returns
    -------
    fig, ax
        Figure and axis objects
    """
    import pypesto.visualize as pypesto_viz

    # Return empty figure if no optimization results
    if not hasattr(result, "optimize_result") or result.optimize_result is None:
        return plot_empty_figure(
            "No optimization results available", kwargs.get("figsize", (10, 6))
        )

    # Get parameters
    figsize = kwargs.pop("figsize", (10, 6))
    scale_y = kwargs.pop("scale_y", "log10")

    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    pypesto_viz.waterfall(results=result, ax=ax, scale_y=scale_y, **kwargs)

    plt.tight_layout()

    return fig, ax


def plot_profiles(result, true_params=None, **kwargs):
    """
    Plot parameter profiles with true parameter values.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing profile results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    **kwargs
        Additional arguments passed to pypesto.visualize.profiles
        figsize: tuple - Figure size
        profile_indices: list of int - Parameter indices to include
        show_bounds: bool - Whether to show parameter bounds

    Returns
    -------
    fig, axes
        Figure and axes objects
    """
    import pypesto.visualize as pypesto_viz

    # Create profile result handler
    profile_result = ProfileResult(result, true_params, kwargs.get("profile_indices"))

    # Return empty figure if no profile results
    if not profile_result.has_profile_results():
        return plot_empty_figure(
            "No profile results available", kwargs.get("figsize", (12, 5))
        )

    # Get plotting parameters
    figsize = kwargs.pop("figsize", (12, 5))
    show_bounds = kwargs.pop("show_bounds", True)

    # Get indices with profile results if not provided
    profile_indices = kwargs.pop("profile_indices", None)
    if profile_indices is None:
        profile_indices = profile_result.get_profile_indices()

    # Create the profiles plot
    axes = pypesto_viz.profiles(
        results=result,
        profile_indices=profile_indices,
        size=figsize,
        show_bounds=show_bounds,
        **kwargs
    )

    # Get the figure
    fig = plt.gcf()

    # Skip true values if not provided
    true_values = profile_result.get_true_parameter_values(scaled=True)
    if not true_values:
        plt.tight_layout()
        return fig, axes

    # Convert axes to list for iteration
    axes_list = (
        [axes]
        if not isinstance(axes, (list, np.ndarray))
        else axes.flatten() if isinstance(axes, np.ndarray) else axes
    )

    # Create true value line style
    true_line = plt.Line2D(
        [0], [0], color="red", linestyle="--", linewidth=2, label="True value"
    )

    # Add vertical lines for true parameter values
    for i, param_idx in enumerate(profile_indices):
        # Skip if out of range
        if i >= len(axes_list):
            continue

        # Get parameter name
        param_name = (
            profile_result.param_names[param_idx]
            if param_idx < len(profile_result.param_names)
            else None
        )

        # Skip if no true value
        if param_name is None or param_name not in true_values:
            continue

        # Get axis and add line
        ax = axes_list[i]
        ax.axvline(
            x=true_values[param_name],
            color="red",
            linestyle="--",
            linewidth=2,
            label="True value",
        )

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        if "True value" not in labels:
            if handles:
                handles.append(true_line)
                ax.legend(handles=handles)
            else:
                ax.legend(handles=[true_line])

    plt.tight_layout()
    return fig, axes


def plot_parameter_traces(result, true_params=None, **kwargs):
    """
    Plot parameter traces from MCMC sampling with true parameter values.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing sampling results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    **kwargs
        Additional arguments passed to pypesto.visualize.sampling_parameter_traces
        figsize: tuple - Figure size
        i_chain: int - Chain index to use
        parameter_indices: list of int - Parameters to include
        stepsize: int - Only include every nth sample
        burn_in: int or float - Samples to discard

    Returns
    -------
    fig, axes
        Figure and axes objects
    """
    import pypesto.visualize as pypesto_viz

    # Create sampling result handler
    sampling_result = SamplingResult(
        result, true_params, kwargs.get("parameter_indices")
    )

    # Return empty figure if no sampling results
    if not sampling_result.has_sampling_results():
        return plot_empty_figure(
            "No sampling results available", kwargs.get("figsize", (12, 5))
        )

    # Get plotting parameters
    figsize = kwargs.pop("figsize", (12, 5))
    i_chain = kwargs.pop("i_chain", 0)
    parameter_indices = kwargs.pop("parameter_indices", None)
    stepsize = kwargs.pop("stepsize", 1)
    burn_in = kwargs.pop("burn_in", 0)

    # Let pypesto create the plot
    axes = pypesto_viz.sampling_parameter_traces(
        result=result,
        i_chain=i_chain,
        par_indices=parameter_indices,
        stepsize=stepsize,
        size=figsize,
        **kwargs
    )

    # Get the figure
    fig = plt.gcf()

    # Skip if no true values
    true_values = sampling_result.get_true_parameter_values(scaled=True)
    if not true_values:
        plt.tight_layout()
        return fig, axes

    # Create a line for the legend
    true_line = plt.Line2D(
        [0], [0], color="red", linestyle="--", linewidth=2, label="True value"
    )

    # Convert axes to list for iteration
    axes_list = (
        [axes]
        if not isinstance(axes, (list, np.ndarray))
        else axes.flatten() if isinstance(axes, np.ndarray) else axes
    )

    # Add true values to each subplot
    for i, param_name in enumerate(sampling_result.param_names):
        # Skip if out of range or no true value
        if i >= len(axes_list) or param_name not in true_values:
            continue

        # Get axis and add horizontal line
        ax = axes_list[i]
        ax.axhline(true_values[param_name], color="red", linestyle="--", linewidth=2)

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        if "True value" not in labels:
            if handles:
                handles.append(true_line)
                ax.legend(handles=handles)
            else:
                ax.legend(handles=[true_line])

    plt.tight_layout()
    return fig, axes


def plot_confidence_intervals(result, true_params=None, **kwargs):
    """
    Plot confidence intervals from MCMC sampling with true parameter values.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing sampling results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    **kwargs
        Additional arguments passed to pypesto.visualize.sampling_parameter_cis
        figsize: tuple - Figure size for the plot

    Returns
    -------
    fig, ax
        Figure and axis objects
    """
    import pypesto.visualize as pypesto_viz

    # Create sampling result handler
    sampling_result = SamplingResult(result, true_params, None)

    # Return empty figure if no results
    if not sampling_result.has_sampling_results():
        return plot_empty_figure(
            "No sampling results available", kwargs.get("figsize", (10, 6))
        )

    # Create figure
    figsize = kwargs.pop("figsize", (10, 6))
    fig, ax = plt.subplots(figsize=figsize)

    # Set default alpha if not provided
    if "alpha" not in kwargs:
        print(kwargs)
        kwargs["alpha"] = [90, 95, 99]  # Default confidence levels

    # Plot confidence intervals
    pypesto_viz.sampling_parameter_cis(result=result, ax=ax, **kwargs)

    # Get true parameter values
    true_values = sampling_result.get_true_parameter_values(scaled=True)
    if not true_values:
        plt.tight_layout()
        return fig, ax

    # Get parameter positions from plot
    y_ticks = ax.get_yticks()
    y_labels = [label.get_text() for label in ax.get_yticklabels()]

    # Collect parameters with true values
    true_x_vals = []
    true_y_pos = []
    for i, param_id in enumerate(y_labels):
        if param_id in true_values:
            true_x_vals.append(true_values[param_id])
            true_y_pos.append(y_ticks[i])

    # Skip if no matching parameters
    if not true_x_vals:
        plt.tight_layout()
        return fig, ax

    # Plot true values
    ax.scatter(
        true_x_vals,
        true_y_pos,
        marker="*",
        s=200,
        color="red",
        label="True value",
        zorder=10,
    )

    # Add to legend
    true_marker = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="red",
        linestyle="none",
        markersize=10,
        label="True value",
    )

    # Update existing legend or create new one
    if ax.get_legend() is not None:
        handles, labels = ax.get_legend_handles_labels()
        if "True value" not in labels:
            handles.append(true_marker)
            ax.legend(handles)
    else:
        ax.legend(handles=[true_marker])

    plt.tight_layout()
    return fig, ax


def plot_optimization_scatter(result, true_params=None, **kwargs):
    """
    Create scatter plot of optimization results with true parameter values.

    Parameters
    ----------
    result : pypesto.Result
        The optimization result object
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    **kwargs
        Additional arguments passed to pypesto.visualize.optimization_scatter
        figsize: tuple - Figure size
        parameter_indices: list or 'free_only' - Parameters to include
        diag_kind: str - Type of diagonal plot ('kde', 'hist', etc.)
        show_bounds: bool - Whether to show parameter bounds

    Returns
    -------
    fig, grid
        Figure and seaborn PairGrid objects
    """
    import pypesto.visualize as pypesto_viz

    # Return empty figure if no optimization results
    if not hasattr(result, "optimize_result") or result.optimize_result is None:
        return plot_empty_figure(
            "No optimization results available", kwargs.get("figsize", (10, 10))
        )

    # Get plotting parameters
    figsize = kwargs.pop("figsize", (10, 10))
    parameter_indices = kwargs.pop("parameter_indices", "free_only")
    diag_kind = kwargs.pop("diag_kind", "kde")
    show_bounds = kwargs.pop("show_bounds", True)

    # Create the scatter plot
    grid = pypesto_viz.optimization_scatter(
        result=result,
        parameter_indices=parameter_indices,
        diag_kind=diag_kind,
        size=figsize,
        show_bounds=show_bounds,
        **kwargs
    )

    # Get the figure and return if no true parameters
    fig = plt.gcf()
    if true_params is None:
        plt.tight_layout()
        return fig, grid

    # Get true parameter values
    opt_result = OptimizationResult(result, true_params, parameter_indices)
    param_names = opt_result.param_names
    true_values = opt_result.get_true_parameter_values(scaled=True)

    # Return if no grid axes or parameter names
    if not param_names or not hasattr(grid, "axes") or len(grid.axes) == 0:
        plt.tight_layout()
        return fig, grid

    # Add true values to each subplot
    for i, param_y in enumerate(param_names):
        if i >= len(grid.axes):
            continue

        for j, param_x in enumerate(param_names):
            if j >= len(grid.axes[i]):
                continue

            # Skip if no true values
            if param_y not in true_values or param_x not in true_values:
                continue

            ax = grid.axes[i, j]

            # Diagonal plots (marginal distributions)
            if i == j:
                ax.axvline(
                    x=true_values[param_y],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="True value",
                )
            # Off-diagonal scatter plots
            else:
                ax.scatter(
                    x=true_values[param_x],
                    y=true_values[param_y],
                    marker="*",
                    s=150,
                    color="red",
                    alpha=0.8,
                    zorder=10,
                    label="True value",
                )

    # Add legend if we have true values
    if true_values:
        true_point = plt.Line2D(
            [0],
            [0],
            marker="*",
            color="red",
            linestyle="none",
            markersize=10,
            label="True value",
        )

        # Add to figure legend if it doesn't exist
        if not plt.figlegend().get_texts():
            plt.figlegend(handles=[true_point], loc="upper right")

    plt.tight_layout()
    return fig, grid


def plot_sampling_scatter(result, true_params=None, **kwargs):
    """
    Create a scatter plot of parameters from MCMC sampling with true values.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing sampling results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    **kwargs
        Additional arguments passed to pypesto.visualize.sampling_scatter
        figsize: tuple - Figure size
        parameter_indices: list of int - Parameters to include

    Returns
    -------
    fig, grid
        Figure and seaborn PairGrid objects
    """
    import pypesto.visualize as pypesto_viz

    # Create sampling result handler
    sampling_result = SamplingResult(
        result, true_params, kwargs.get("parameter_indices")
    )

    # Return empty figure if no sampling results
    if not sampling_result.has_sampling_results():
        return plot_empty_figure(
            "No sampling results available", kwargs.get("figsize", (10, 10))
        )

    # Get parameters
    figsize = kwargs.pop("figsize", (10, 10))

    # Create scatter plot
    grid = pypesto_viz.sampling_scatter(result=result, size=figsize, **kwargs)

    # Get figure and true values
    fig = plt.gcf()
    true_values = sampling_result.get_true_parameter_values(scaled=True)

    # Return if no true values or no valid grid
    if not true_values or not hasattr(grid, "axes") or len(grid.axes) == 0:
        plt.tight_layout()
        return fig, grid

    param_names = grid.x_vars

    # Add true values to plots
    for i, param_y in enumerate(param_names):
        if i >= len(grid.axes):
            continue

        for j, param_x in enumerate(param_names):
            if j >= len(grid.axes[i]):
                continue

            # Skip if no true values
            if param_y not in true_values or param_x not in true_values:
                continue

            ax = grid.axes[i, j]

            # Diagonal plots (marginals)
            if i == j:
                ax.axvline(
                    x=true_values[param_y],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="True value",
                )
            # Off-diagonal scatter plots
            else:
                ax.scatter(
                    x=true_values[param_x],
                    y=true_values[param_y],
                    marker="*",
                    s=150,
                    color="red",
                    alpha=0.8,
                    zorder=10,
                    label="True value",
                )

    # Add legend
    true_point = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="red",
        linestyle="none",
        markersize=10,
        label="True value",
    )

    # Add to figure if no legend exists
    if not plt.figlegend().get_texts():
        plt.figlegend(handles=[true_point], loc="upper right")

    plt.tight_layout()
    return fig, grid


def visualize_parameter_estimation(result, true_params=None, **kwargs):
    """
    Create a comprehensive visualization of parameter estimation results.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing estimation results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    **kwargs
        Additional arguments:
        plots: list or tuple - Which plots to generate
          'waterfall', 'scatter', 'profiles', 'traces', 'intervals', 'sampling_scatter'
        figsize: tuple - Base figure size
        waterfall_kwargs: dict - Args for waterfall plot
        scatter_kwargs: dict - Args for optimization scatter plot
        profiles_kwargs: dict - Args for profile plots
        traces_kwargs: dict - Args for trace plots
        intervals_kwargs: dict - Args for confidence interval plots
        sampling_scatter_kwargs: dict - Args for sampling scatter plot

    Returns
    -------
    dict
        Dictionary mapping plot types to (fig, ax) tuples
    """
    # Extract plotting parameters
    plots = kwargs.pop(
        "plots", ("waterfall", "scatter", "profiles", "traces", "intervals")
    )
    figsize = kwargs.pop("figsize", (10, 6))

    # Get plot-specific kwargs
    plot_kwargs = {
        "waterfall": kwargs.pop("waterfall_kwargs", {}),
        "scatter": kwargs.pop("scatter_kwargs", {}),
        "profiles": kwargs.pop("profiles_kwargs", {}),
        "traces": kwargs.pop("traces_kwargs", {}),
        "intervals": kwargs.pop("intervals_kwargs", {}),
        "sampling_scatter": kwargs.pop("sampling_scatter_kwargs", {}),
    }

    # Initialize results
    results = {}

    # Define plot creators with custom sizes
    plot_creators = {
        "waterfall": lambda: plot_waterfall(
            result=result, figsize=figsize, **plot_kwargs["waterfall"]
        ),
        "scatter": lambda: plot_optimization_scatter(
            result=result,
            true_params=true_params,
            figsize=(figsize[0] * 1.5, figsize[1] * 1.5),
            **plot_kwargs["scatter"]
        ),
        "profiles": lambda: plot_profiles(
            result=result,
            true_params=true_params,
            figsize=(figsize[0] * 1.5, figsize[1]),
            **plot_kwargs["profiles"]
        ),
        "traces": lambda: plot_parameter_traces(
            result=result,
            true_params=true_params,
            figsize=(figsize[0] * 1.5, figsize[1]),
            **plot_kwargs["traces"]
        ),
        "intervals": lambda: plot_confidence_intervals(
            result=result,
            true_params=true_params,
            figsize=figsize,
            **plot_kwargs["intervals"]
        ),
        "sampling_scatter": lambda: plot_sampling_scatter(
            result=result,
            true_params=true_params,
            figsize=(figsize[0] * 1.5, figsize[1] * 1.5),
            **plot_kwargs["sampling_scatter"]
        ),
    }

    # Create requested plots if data is available
    for plot_type in plots:
        # Skip if plot type not defined
        if plot_type not in plot_creators:
            continue

        # Check data availability based on plot type
        if plot_type in ["waterfall", "scatter"] and not hasattr(
            result, "optimize_result"
        ):
            continue
        if plot_type == "profiles" and not hasattr(result, "profile_result"):
            continue
        if plot_type in ["traces", "intervals", "sampling_scatter"] and not hasattr(
            result, "sample_result"
        ):
            continue

        # Create the plot
        results[plot_type] = plot_creators[plot_type]()

    return results
