import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pypesto
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Optional

from pypesto.sample import calculate_ci_mcmc_sample

from polypesto.core._study import SimulatedExperimentDict, ResultsDict, Study


def extract_fA0_value(cond_id: str, num_values: int = 1) -> float:

    match = re.search(r"fA0_\[([^\]]+)\]", cond_id)
    if not match:
        print(f"Condition ID {cond_id} does not match the expected format.")
        return None

    fA0_str = match.group(1)
    if num_values == 1:
        return float(fA0_str)

    return [float(i) for i in fA0_str.split(",")]


from polypesto.core.results import SamplingResult


def plot_comparisons_1D(
    study: Study, p_id: str, axes: Optional[List[Axes]] = None
) -> List[Axes]:

    exp = study.get_experiments(filter_p_id=p_id)
    results = study.get_results(p_id)

    # Get any experiment from exp
    experiment = list(exp.values())[0]
    param_names = experiment.petab_problem.get_optimization_parameters()

    if axes is None:
        fig, axes = plt.subplots(1, len(param_names), figsize=(12, 4))

    colors = ["#60A88D", "#2D69B2"]
    for j, ((cond_id, p_id), experiment) in enumerate(exp.items()):

        print(experiment.petab_problem.x_free_ids)
        print(experiment.petab_problem.get_optimization_parameters())

        print(f"Plotting comparison for condition {cond_id}, parameter set {p_id}")
        result = results[(cond_id, p_id)]
        true_params = experiment.true_params.to_dict()

        sampling_result = SamplingResult(
            result,
            true_params=true_params,
            parameter_indices=experiment.petab_problem.x_free_indices,
        )
        alpha = 0.95
        intervals = sampling_result.get_credible_intervals(
            alpha_levels=(alpha,), burn_in=0
        )
        print(intervals)

        fA0_value = extract_fA0_value(cond_id, num_values=1)

        for i, p_name in enumerate(param_names):

            if p_name not in intervals:
                print(f"Parameter {p_name} not found in intervals.")
                continue

            lb, med, ub = intervals[p_name][alpha]
            lb, med, ub = np.log10(lb), np.log10(med), np.log10(ub)

            lb = np.abs(med - lb)
            ub = np.abs(ub - med)

            print(f"Parameter {p_name} confidence intervals: {lb}, {med}, {ub}")

            axes[i].errorbar(
                fA0_value,
                med,
                yerr=[[lb], [ub]],
                fmt="o",
                color=colors[i],
                capsize=5,
                label=f"{p_id}" if j == 0 else None,
            )

            if j == 0:
                plot_params = np.log10([true_params[p_name]])
                axes[i].axhline(plot_params, color="red", linestyle="--")
                axes[i].set_ylim([-2, 2])

    return axes


# def plot_comparisons_1D_fill(
#     study: Study, p_id: str, axes: Optional[List[Axes]] = None
# ) -> List[Axes]:
#     exp = study.get_experiments(filter_p_id=p_id)
#     results = study.get_results(p_id)

#     # Get any experiment from exp
#     experiment = list(exp.values())[0]
#     param_names = experiment.petab_problem.get_optimization_parameters()

#     if axes is None:
#         fig, axes = plt.subplots(1, len(param_names), figsize=(12, 4))

#     # If only one parameter, make axes a list
#     if len(param_names) == 1:
#         axes = [axes]

#     # Dictionary to hold data for each parameter
#     param_data = {
#         p_name: {"fA0": [], "median": [], "lower": [], "upper": []}
#         for p_name in param_names
#     }

#     # Collect data for all fA0 values
#     for (cond_id, curr_p_id), experiment in exp.items():
#         # Skip if not the requested parameter set
#         if curr_p_id != p_id:
#             continue

#         fA0_value = extract_fA0_value(cond_id)
#         result = results[(cond_id, curr_p_id)]

#         true_params = experiment.true_params.to_dict()
#         sampling_result = SamplingResult(
#             result,
#             true_params=true_params,
#             parameter_indices=experiment.petab_problem.x_free_indices,
#         )

#         alpha = 0.95
#         intervals = sampling_result.get_credible_intervals(
#             alpha_levels=(alpha,), burn_in=0
#         )

#         # Store values for each parameter
#         for p_name in param_names:
#             if p_name not in intervals:
#                 print(f"Parameter {p_name} not found in intervals.")
#                 continue

#             lb, med, ub = intervals[p_name][alpha]
#             # Convert to log scale
#             lb, med, ub = np.log10(lb), np.log10(med), np.log10(ub)

#             param_data[p_name]["fA0"].append(fA0_value)
#             param_data[p_name]["median"].append(med)
#             param_data[p_name]["lower"].append(lb)
#             param_data[p_name]["upper"].append(ub)

#     # Colors for each parameter
#     colors = {"rA": "blue", "rB": "green"}

#     # Plot data for each parameter
#     for i, p_name in enumerate(param_names):
#         data = param_data[p_name]

#         # Sort all data by fA0 value (needed for fill_between)
#         sorted_indices = np.argsort(data["fA0"])
#         fA0_sorted = [data["fA0"][i] for i in sorted_indices]
#         median_sorted = [data["median"][i] for i in sorted_indices]
#         lower_sorted = [data["lower"][i] for i in sorted_indices]
#         upper_sorted = [data["upper"][i] for i in sorted_indices]

#         color = colors.get(
#             p_name, f"C{i}"
#         )  # Use predefined color or fallback to default

#         # Plot median line with markers at actual data points
#         axes[i].plot(fA0_sorted, median_sorted, "o-", color=color, label=p_name)

#         # Add confidence interval as shaded area
#         axes[i].fill_between(
#             fA0_sorted,
#             lower_sorted,
#             upper_sorted,
#             color=color,
#             alpha=0.3,  # Semi-transparent
#         )

#         # Add true parameter value if available
#         if true_params and p_name in true_params:
#             true_value = np.log10(true_params[p_name])
#             axes[i].axhline(true_value, color="red", linestyle="--", label="True Value")

#         # Set labels and title
#         axes[i].set_xlabel("fA0 Value")
#         axes[i].set_ylabel(f"log10({p_name})")
#         axes[i].set_title(f"Parameter {p_name} (Set {p_id})")
#         axes[i].legend()
#         axes[i].grid(True, alpha=0.3)
#         axes[i].set_ylim([-2, 2])

#     return axes


# def plot_all_comparisons_1D_fill(study: Study) -> None:
#     parameter_ids = study.get_parameter_ids()
#     print(f"Plotting {len(parameter_ids)} comparisons...")

#     for i, p_id in enumerate(parameter_ids):
#         print(f"Plotting comparison for parameter set {i + 1}/{len(parameter_ids)}")

#         # Create a new figure for each parameter set
#         fig, axes = plt.subplots(
#             1, 2, figsize=(12, 4)
#         )  # Assuming 2 parameters (rA, rB)

#         # Plot with confidence intervals
#         ax = plot_comparisons_1D_fill(study, p_id, axes)

#         # Save the figure
#         comp_dir = (
#             list(study.get_experiments(filter_p_id=p_id).values())[
#                 0
#             ].experiment.paths.base_dir.parent.absolute()
#             / "comparison_plots"
#         )
#         comp_dir.mkdir(parents=True, exist_ok=True)
#         fig_path = comp_dir / f"comparison_plot_CI_{p_id}.png"
#         fig.tight_layout()
#         fig.savefig(fig_path, bbox_inches="tight", dpi=300)
#         plt.close(fig)


def plot_comparisons_1D_fill(
    study: Study, p_id: str, ax: Optional[Axes] = None
) -> Axes:
    exp = study.get_experiments(filter_p_id=p_id)
    results = study.get_results(p_id)

    # Get any experiment from exp
    experiment = list(exp.values())[0]
    param_names = experiment.petab_problem.get_optimization_parameters()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Square figure

    # Dictionary to hold data for each parameter
    param_data = {
        p_name: {"fA0": [], "median": [], "lower": [], "upper": []}
        for p_name in param_names
    }

    # Collect data for all fA0 values
    for (cond_id, curr_p_id), experiment in exp.items():
        # Skip if not the requested parameter set
        if curr_p_id != p_id:
            continue

        fA0_value = extract_fA0_value(cond_id)
        result = results[(cond_id, curr_p_id)]

        true_params = experiment.true_params.to_dict()
        sampling_result = SamplingResult(
            result,
            true_params=true_params,
            parameter_indices=experiment.petab_problem.x_free_indices,
        )

        alpha = 0.95
        intervals = sampling_result.get_credible_intervals(
            alpha_levels=(alpha,), burn_in=0
        )

        # Store values for each parameter
        for p_name in param_names:
            if p_name not in intervals:
                print(f"Parameter {p_name} not found in intervals.")
                continue

            lb, med, ub = intervals[p_name][alpha]

            # Store raw values (not log10 transformed)
            param_data[p_name]["fA0"].append(fA0_value)
            param_data[p_name]["median"].append(med)
            param_data[p_name]["lower"].append(lb)
            param_data[p_name]["upper"].append(ub)

    # Colors for each parameter
    colors = {"rA": "#60A88D", "rB": "#2D69B2"}  # Professional colors
    # Plot data for each parameter
    for i, p_name in enumerate(param_names):
        data = param_data[p_name]

        # Sort all data by fA0 value (needed for fill_between)
        sorted_indices = np.argsort(data["fA0"])
        fA0_sorted = [data["fA0"][i] for i in sorted_indices]
        median_sorted = [data["median"][i] for i in sorted_indices]
        lower_sorted = [data["lower"][i] for i in sorted_indices]
        upper_sorted = [data["upper"][i] for i in sorted_indices]

        color = colors.get(p_name, f"C{i}")

        # Plot median line with markers at actual data points
        ax.plot(
            fA0_sorted,
            median_sorted,
            "o-",
            color=color,
            label=p_name,
            linewidth=2,
            markersize=6,
        )

        # Add confidence interval as shaded area
        ax.fill_between(
            fA0_sorted,
            lower_sorted,
            upper_sorted,
            color=color,
            alpha=0.2,  # Subtle transparency
        )

        # Add true parameter value if available
        if true_params and p_name in true_params:
            true_value = true_params[p_name]
            ax.axhline(
                true_value, color=color, linestyle="--", alpha=0.7, linewidth=1.5
            )

    # Set up log scale for y-axis
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_ylim([0.01, 100])
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    # Clean up the plot for manuscript quality
    # ax.spines["top"].set_visible(True)
    # ax.spines["right"].set_visible(True)
    # ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(
        axis="both",       # Both x and y axes
        which="major",     # Only major ticks
        direction="in",    # Make ticks point inward
        labelsize=10,      # Tick label font size
        top=True,          # Show ticks on top x-axis
        right=True,        # Show ticks on right y-axis
        length=6           # Length of tick marks
    )
    # Add minor ticks and make them face inwards too
    ax.tick_params(
        axis="both",
        which="minor",     # Minor ticks
        direction="in",    # Make ticks point inward
        top=True,          # Show minor ticks on top x-axis
        right=True,        # Show minor ticks on right y-axis
        length=3           # Shorter length for minor ticks
)

    # Set labels and title
    ax.set_xlabel("Feed Fraction of A")
    # ax.set_xlabel("$f_{A0}$ Value", fontsize=12)
    # ax.set_ylabel("Parameter Value", fontsize=12)
    # ax.set_title(f"Parameter Set {p_id}", fontsize=14)

    # Add legend
    # ax.legend(frameon=False, loc="upper right")

    return ax


def plot_all_comparisons_1D_fill(study: Study) -> None:
    parameter_ids = study.get_parameter_ids()
    print(f"Plotting {len(parameter_ids)} comparisons...")

    for i, p_id in enumerate(parameter_ids):
        print(f"Plotting comparison for parameter set {i + 1}/{len(parameter_ids)}")

        # Create a new square figure for each parameter set
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=500)

        # Plot with confidence intervals
        plot_comparisons_1D_fill(study, p_id, ax)

        # Save the figure
        comp_dir = (
            list(study.get_experiments(filter_p_id=p_id).values())[
                0
            ].experiment.paths.base_dir.parent.absolute()
            / "comparison_plots"
        )
        comp_dir.mkdir(parents=True, exist_ok=True)
        fig_path = comp_dir / f"comparison_plot_CI_{p_id}.png"
        fig.tight_layout()
        fig.savefig(fig_path, bbox_inches="tight", dpi=500)
        plt.close(fig)


def plot_all_comparisons_1D(study: Study) -> None:
    parameter_ids = study.get_parameter_ids()

    print(f"Plotting {len(parameter_ids)} comparisons...")
    for i, p_id in enumerate(parameter_ids):
        print(f"Plotting comparison for parameter set {i + 1}/{len(parameter_ids)}")
        experiments = study.get_experiments(filter_p_id=p_id)

        print(f"Found {len(experiments)} experiments for parameter set {p_id}")
        results = study.get_results(p_id)

        # plot_comparisons(experiments, results, axes)
        ax = plot_comparisons_1D(study, p_id)
        comp_dir = (
            list(experiments.values())[0].experiment.paths.base_dir.parent.absolute()
            / "comparison_plots"
        )
        comp_dir.mkdir(parents=True, exist_ok=True)
        fig_path = comp_dir / f"comparison_plot_{p_id}.png"
        
        fig = plt.gcf()
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        # break


def plot_comparisons_2D(study: Study, p_id: str) -> None:
    exp = study.get_experiments(filter_p_id=p_id)
    results = study.get_results(p_id)
    param_names = ["rA", "rB"]

    # Extract all unique fA01 and fA02 values to create the grid
    fA01_values = []
    fA02_values = []
    parameter_data = {}

    # First pass: collect all data points
    for (cond_id, p_id), experiment in exp.items():
        try:
            fA01, fA02 = extract_fA0_value(cond_id, num_values=2)
            fA01_values.append(fA01)
            fA02_values.append(fA02)

            result = results[(cond_id, p_id)]
            true_params = experiment.true_params.to_dict()

            sampling_result = SamplingResult(result, true_params=true_params)
            alpha = 0.95
            intervals = sampling_result.get_credible_intervals(
                alpha_levels=(alpha,), burn_in=0
            )

            for p_name in param_names:
                if p_name in intervals:
                    lb, med, ub = intervals[p_name][alpha]
                    # Convert to log scale
                    lb, med, ub = np.log10(lb), np.log10(med), np.log10(ub)

                    # Calculate deviation from true value
                    true_value = np.log10(true_params[p_name])
                    deviation = med - true_value

                    # Calculate confidence interval width
                    ci_width = ub - lb

                    # Store data
                    if p_name not in parameter_data:
                        parameter_data[p_name] = {}
                    parameter_data[p_name][(fA01, fA02)] = {
                        "deviation": deviation,
                        "ci_width": ci_width,
                    }
        except Exception as e:
            print(f"Error processing condition {cond_id}: {str(e)}")

    # Get unique sorted values for grid
    fA01_values = sorted(set(fA01_values))
    fA02_values = sorted(set(fA02_values))

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Common mesh grid for all plots
    X, Y = np.meshgrid(fA01_values, fA02_values)

    # Create heatmaps for each parameter and metric
    for col, p_name in enumerate(param_names):
        # Initialize data arrays
        Z_deviation = np.zeros_like(X, dtype=float)
        Z_ci_width = np.zeros_like(X, dtype=float)

        # Fill in the data
        for i, fA01 in enumerate(fA01_values):
            for j, fA02 in enumerate(fA02_values):
                if (fA01, fA02) in parameter_data[p_name]:
                    Z_deviation[j, i] = parameter_data[p_name][(fA01, fA02)][
                        "deviation"
                    ]
                    Z_ci_width[j, i] = parameter_data[p_name][(fA01, fA02)]["ci_width"]
                else:
                    Z_deviation[j, i] = np.nan
                    Z_ci_width[j, i] = np.nan

        # Determine symmetric colormap range for deviation (centered at 0)
        max_abs_dev = np.nanmax(np.abs(Z_deviation))
        dev_vmin, dev_vmax = -max_abs_dev, max_abs_dev

        # Plot deviation from true value (Row 0)
        im1 = axes[0, col].pcolormesh(
            X,
            Y,
            Z_deviation,
            shading="auto",
            cmap="RdBu_r",  # Red-Blue diverging colormap centered at 0
            vmin=dev_vmin,
            vmax=dev_vmax,
        )
        axes[0, col].set_title(f"{p_name} Deviation from True Value")
        fig.colorbar(im1, ax=axes[0, col])

        # Plot confidence interval width (Row 1)
        im2 = axes[1, col].pcolormesh(
            X, Y, Z_ci_width, shading="auto", cmap="viridis"
        )  # Sequential colormap for width
        axes[1, col].set_title(f"{p_name} CI Width")
        fig.colorbar(im2, ax=axes[1, col])

        # Add axis labels
        for row in range(2):
            axes[row, col].set_xlabel("fA01")
            axes[row, col].set_ylabel("fA02")

    # Add global title
    fig.suptitle(f"Parameter Estimation Results for {p_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    # Save the figure
    comp_dir = (
        list(exp.values())[0].experiment.paths.base_dir.parent.absolute()
        / "comparison_plots"
    )
    comp_dir.mkdir(parents=True, exist_ok=True)
    fig_path = comp_dir / f"2D_grid_comparison_{p_id}.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# Function to call for all parameter sets
def plot_all_comparisons_2D(study: Study) -> None:
    parameter_ids = study.get_parameter_ids()

    print(f"Plotting {len(parameter_ids)} 2D grid comparisons...")
    for i, p_id in enumerate(parameter_ids):
        print(
            f"Plotting 2D grid comparison for parameter set {i + 1}/{len(parameter_ids)}"
        )
        plot_comparisons_2D(study, p_id)


def summarize_study(study: Study) -> pd.DataFrame:

    # parameter_ids = study.get_parameter_ids()
    for (cond_id, p_id), sim_exp in study.experiments.items():

        true_params = sim_exp.true_params.to_dict()
        exp = sim_exp.experiment

        result = study.results[(cond_id, p_id)]

        sampling_result = SamplingResult(result, true_params=true_params)
        alpha = 0.95
        intervals = sampling_result.get_credible_intervals(
            alpha_levels=(alpha,), burn_in=0
        )

        # lb, med, ub =

    return


"""
DataFrame columns:

p_id: str
- e.g. "p_000"

true_params: dict
- e.g. {
    "rA": 0.1, 
    "rB": 0.5
}

cond_id: str
- e.g. "fA0_[0.1]_cM0_[1.0]"

conditions: dict
- e.g. {
    "fA0": [0.1, 0.2],
    "cM0": [1.0, 2.0]
}

sampling_confidence_intervals: dict
- e.g. {
    "rA": {
        0.95: (0.05, 0.5, 0.95)
    },
    "rB": {
        0.95: (0.05, 0.5, 0.95)
    }
}


"""
