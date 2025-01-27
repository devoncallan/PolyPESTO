import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.cm as cm
import matplotlib.colors as colors
from typing import List, Dict, Any, Union, Tuple, Optional
import petab.v1.C as C


# Function for plotting all measurements data
def plot_all_measurements(
    meas_df: pd.DataFrame,
    group_by: str = C.CONDITION_ID,
    axes: List[Axes] = None,
    format_axes_kwargs: Dict[str, Any] = None,
    plot_style: str = "lines",
    **kwargs,
) -> List[Axes]:
    """
    Plot measurement data grouped by either conditions or observables across multiple subplots.

    Parameters
    ----------
    meas_df : pd.DataFrame
        A pandas DataFrame containing measurement data. The DataFrame is expected to have columns:
        - C.CONDITION_ID, C.OBSERVABLE_ID, C.TIME, C.MEASUREMENT
    group_by : str, default=C.CONDITION_ID
        Column to group subplots by. Must be either:
        - C.CONDITION_ID: Creates a subplot for each condition, showing all observables within each panel.
        - C.OBSERVABLE_ID: Creates a subplot for each observable, showing all conditions within each panel.
        Raises a ValueError if an invalid option is provided.
    axes : List[Axes], optional
        A list of matplotlib Axes objects to use for plotting. If not provided, subplots are created automatically.
        If provided, the number of Axes must >= the number of panels (conditions or observables).
    format_axes_kwargs : Dict[str, Any], optional
        A dictionary of matplotlib Axes formatting methods and their arguments.
        Example: `{"set_xlabel": "Time", "set_ylabel": "Measurement", "set_xlim": (0, 1)}`.
    plot_style : {"lines", "scatter", "both"}, default="lines"
        The style to use for plotting:
        - `"lines"`: Solid lines only.
        - `"scatter"`: Markers only, no connecting lines.
        - `"both"`: Both solid lines and markers.
    **kwargs : dict
        Additional keyword arguments passed to the plotting function.

    Returns
    -------
    List[Axes]
        A list of matplotlib Axes objects.

    Raises
    ------
    ValueError
        - If `group_by` is not one of `C.CONDITION_ID` or `C.OBSERVABLE_ID`.
        - If `axes` is provided but its length does not match the expected number of subplots.

    Notes
    -----
    - This function divides the data into subplots based on the `group_by` column. Each subplot represents either
      a condition (with multiple observables) or an observable (with multiple conditions).
    - The `get_plot_formatting` function is used to dynamically assign a unique color, marker, and linestyle
      to each (condition, observable) pair, ensuring consistent formatting.
    - The plotting is performed by the `plot_measurements` function, which handles the individual data subsets.

    Example
    -------
    >>> axes = plot_all_measurements(
    ...     meas_df=df,
    ...     group_by=C.CONDITION_ID,
    ...     plot_style="both",
    ...     format_axes_kwargs={
    ...         "set_xlabel": "Time",
    ...         "set_ylabel": "Measurement",
    ...         "set_xlim": (0, 10),
    ...         "set_ylim": (0, 1),
    ...     },
    ...     alpha=0.8
    ... )
    >>> plt.tight_layout()
    >>> plt.show()
    """

    MAX_NUM_ROWS = 4

    conditions = meas_df[C.CONDITION_ID].unique()
    observables = meas_df[C.OBSERVABLE_ID].unique()

    if group_by not in ['C.CONDITION_ID','C.OBSERVABLE']:
        raise ValueError('Invalid group_by value passed in.')
    
    if group_by is 'C.CONDITION_ID':
        num_panels = len(conditions)
    else:
        num_panels = len(observables)

    if axes is None:
        fig, axes_list = plt.subplots(1, num_panels)
        axes_list = axes.flatten() 
    else:
        axes_list = axes

    formatting = get_plot_formatting(conditions, observables, plot_style=plot_style)

    for i, condition in enumerate(conditions):
        for j, observable in enumerate(observables):
            data = meas_df[(meas_df[C.CONDITION_ID] == condition) & (meas_df[C.OBSERVABLE_ID] == observable)]

            if group_by is 'C.CONDITION_ID':
                ax = axes_list[i]
                label = observable
            else:
                ax = axes_list[j]
                label = condition

            color, marker, linestyle = formatting[(observable, condition)]

            plot_measurements(ax, data, label=label, color=color, marker=marker, linestyle=linestyle)

    format_axes(axes_list, **kwargs)

    return axes_list

    # Parse the unique conditions and observables
    # group_by must be either C.CONDITION_ID or C.OBSERVABLE_ID (raise ValueError otherwise)
    # Determine num_panels (num conditions or observables) based on group_by
    # If axes is not provided, create a figure with subplots (with MAX_NUM_ROWS)
    
    # Assign plot formatting based on observables, conditions, and plot_style (get_plot_formatting)

    # Loop through each condition:
    #    Loop through each observable:
    #       Get subset of dataframe for given condition/observable
    #       Get the relevant axes (based on group_by) to plot the data
    #       Get the label (condition or observable, based on group_by)
    #       Get color, marker, and linestyle from above
    #       Plot data. Pass in color, marker, linestyle, AND **kwargs

    # Format the axes based on the provided format_axes_kwargs if format_axes_kwargs is not None

    # return axes


# Function for plotting a single figure
def plot_measurements(ax: Axes, meas_df: pd.DataFrame, **kwargs):
    """
    Plot measurement data on a single Axes object using the specified styles
    and any additional keyword arguments.

    Parameters
    ----------
    ax : Axes
        A matplotlib Axes object where the data will be plotted.
    meas_df : pd.DataFrame
        A pandas DataFrame containing the data to plot. Must have columns:
        - C.TIME: Time values for the measurements.
        - C.MEASUREMENT: Measurement values to plot.
    **kwargs : dict
        Additional keyword arguments for plot customization.

    Returns
    -------
    None
        This function modifies the provided Axes object but does not return a value.

    Example
    -------
    >>> plot_measurements(
    ...     ax, meas_df=df, color="blue", marker="o", linestyle="-", label="Sample Data"
    ... )
    """

    x = meas_df[C.TIME]
    y = meas_df[C.MEASUREMENT]
    ax.plot(x, y, **kwargs)

    # ax.set_xlabel(in_kwargs(customization, 'set_xlabel'))
    # ax.set_ylabel(in_kwargs(customization, 'set_ylabel'))
    # ax.set_xlim(in_kwargs(customization, 'set_xlim'))
    # ax.set_ylim(in_kwargs(customization, 'set_ylim'))
    # ax.set_title(in_kwargs(customization, 'set_title'))    

# def in_kwargs(customization: Dict, value: str):
#     default = {'color': 'red',
#                'marker': None,
#                'linestyle': '',
#                'label': None,
#                'set_xlabel': 'Time',
#                'set_ylabel': 'Measurement',
#                'set_xlim': (0, 1),
#                'set_ylim': (0, 1),
#                'set_title': 'Plot of Measurements over Time'}

#     if value in customization:
#         ret = customization[value]
#     else:
#         ret = default[value]

#     return ret


def get_color_shades(colormap_name: str, n_shades: int) -> List[str]:
    """
    Generate color shades based on a given base colormap.

    Parameters
    ----------
    colormap_name : str
        Name of the base colormap (e.g., "Blues", "Reds", "Greens").
        Must be a valid colormap recognized by matplotlib.
    n_shades : int
        Number of distinct shades to generate.

    Returns
    -------
    List[str]
        A list of color codes in the hex format.

    Raises
    ------
    ValueError
        If the provided `colormap_name` is invalid or not recognized by
        matplotlib.
    """
    try:
        # Create the colormap with n_shades + 1 levels, then skip the first one
        # (often white or very light for some colormaps).
        colormap = cm.get_cmap(colormap_name, n_shades + 1)
        return [colors.to_hex(colormap(i)) for i in range(1, n_shades + 1)]
    except Exception as e:
        raise ValueError(f"Invalid colormap: {colormap_name}") from e


def get_plot_formatting(
    observables: List[str], conditions: List[str], plot_style: str = "lines"
) -> Dict[Tuple[str, str], Tuple[str, str, str]]:
    """
    Assign a unique color, marker, and linestyle combination for each
    (observable, condition) pair. The colors are generated using different
    base colormaps, markers cycle through a predefined list, and linestyles
    are determined by the plot style.

    Parameters
    ----------
    observables : List[str]
        A list of unique observables.
    conditions : List[str]
        A list of unique conditions.
    plot_style : {"lines", "scatter", "both"}, default="lines"
        The desired style for plotting:
        - "lines": Linestyle is solid, no marker.
        - "scatter": Marker only, no linestyle.
        - "both": Both marker and solid line.

    Returns
    -------
    Dict[Tuple[str, str], Tuple[str, str, str]]
        A dictionary mapping (observable, condition) pairs to a tuple of
        (color, marker, linestyle). The values depend on the selected plot style.

    Notes
    -----
    - Colors are generated from distinct colormaps for each observable.
    - Markers cycle through a predefined list of marker styles.
    - Linestyles are adjusted based on the provided plot style.
    """
    markers = ["o", "s", "^", "v", "D", "*", "P", "X"]
    colormap_names = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys"]

    format_dict = {}

    # Determine linestyle based on plot_style
    if plot_style == "lines":
        linestyle = "-"
    elif plot_style == "scatter":
        linestyle = "None"
    elif plot_style == "both":
        linestyle = "-"
    else:
        raise ValueError(f"Unknown plot_style: {plot_style}")

    # For each (obs, cond) pair, assign a color, marker, and linestyle
    for i, obs in enumerate(observables):
        obs_colormap_name = colormap_names[i % len(colormap_names)]
        # Generate enough color shades for all conditions
        obs_colormap = get_color_shades(obs_colormap_name, len(conditions))

        for j, cond in enumerate(conditions):
            obs_color = obs_colormap[j]
            # Use markers for scatter or both styles
            obs_marker = (
                markers[j % len(markers)] if plot_style in ["scatter", "both"] else None
            )
            format_dict[(obs, cond)] = (obs_color, obs_marker, linestyle)

    return format_dict


def format_axes(axes: Union[Axes, List[Axes]], **format_kwargs: Any) -> None:
    """
    Format one or multiple matplotlib Axes objects using dynamically
    called methods passed through keyword arguments.

    Parameters
    ----------
    axes : Union[Axes, List[Axes]]
        Single Axes object or a list of Axes objects to format.
    format_kwargs : Dict[str, Any]
        Dictionary of method-value pairs, where each key is an Axes method
        (e.g., 'set_xlabel', 'set_ylabel') and each value is the argument
        to be passed to that method.

    Raises
    ------
    ValueError
        If a specified method does not exist on the Axes object or
        if it exists but is not callable.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> format_axes(ax, set_xlabel="Time (s)", set_ylabel="Amplitude")
    """
    # Ensure we have a list of axes
    if isinstance(axes, Axes):
        axes = [axes]

    # Apply formatting to each axes
    for ax in axes:
        for method, value in format_kwargs.items():
            if not hasattr(ax, method):
                raise ValueError(f"Axes does not have method: '{method}'")
            format_func = getattr(ax, method)
            if not callable(format_func):
                raise ValueError(f"Method '{method}' is not callable on Axes.")
            format_func(value)
