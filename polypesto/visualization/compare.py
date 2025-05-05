import re
import numpy as np
import matplotlib.pyplot as plt
import pypesto
from matplotlib.axes import Axes
from typing import Dict, List, Tuple

from pypesto.sample import calculate_ci_mcmc_sample

from polypesto.core.study import SimulatedExperimentDict, ResultsDict, Study


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


def plot_comparisons_1D(study: Study, p_id: str, axes: List[Axes]) -> List[Axes]:

    exp = study.get_experiments(filter_p_id=p_id)
    results = study.get_results(p_id)

    param_names = ["rA", "rB"]

    for j, ((cond_id, p_id), experiment) in enumerate(exp.items()):

        print(f"Plotting comparison for condition {cond_id}, parameter set {p_id}")
        result = results[(cond_id, p_id)]
        true_params = experiment.true_params.to_dict()

        sampling_result = SamplingResult(result, true_params=true_params)
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
            # lb, med, ub = 10**lb, 10**med, 10**ub
            # lb, med, ub

            axes[i].errorbar(
                fA0_value,
                med,
                yerr=[[lb], [ub]],
                fmt="o",
                capsize=5,
                label=f"{p_id}" if j == 0 else None,
            )

            if j == 0:
                plot_params = np.log10([true_params[p_name]])
                axes[i].axhline(plot_params, color="red", linestyle="--")

    return axes


def plot_all_comparisons_1D(study: Study) -> None:
    parameter_ids = study.get_parameter_ids()

    print(f"Plotting {len(parameter_ids)} comparisons...")
    for i, p_id in enumerate(parameter_ids):
        print(f"Plotting comparison for parameter set {i + 1}/{len(parameter_ids)}")
        experiments = study.get_experiments(filter_p_id=p_id)

        print(f"Found {len(experiments)} experiments for parameter set {p_id}")
        results = study.get_results(p_id)

        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.set_xlim(0, 1)

        # plot_comparisons(experiments, results, axes)
        plot_comparisons_1D(study, p_id, axes)
        comp_dir = (
            list(experiments.values())[0].experiment.paths.base_dir.parent.absolute()
            / "comparison_plots"
        )
        comp_dir.mkdir(parents=True, exist_ok=True)
        fig_path = comp_dir / f"comparison_plot_{p_id}.png"
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
