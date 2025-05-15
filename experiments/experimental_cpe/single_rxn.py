import numpy as np
import matplotlib.pyplot as plt
import os

from polypesto.core.experiment import Experiment
from polypesto.core.experiment.paths import ExperimentPaths
from polypesto.models.CRP2 import ReversibleCPE
from polypesto.core.experiment import run_parameter_estimation
from create_petab import create_tsvs

from polypesto.visualization import (
    plot_optimization_scatter,
    plot_sampling_scatter,
    plot_confidence_intervals,
    plot_waterfall,
    plot_parameter_traces,
    plot_profiles,
    plot_all_measurements,
)

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "data")


monomer_As = [f"{csv_path}/monomer_A.csv", f"{csv_path}/monomer_A_2.csv"]
monomer_Bs = [f"{csv_path}/monomer_B.csv", f"{csv_path}/monomer_B_2.csv"]
condition_ids = ["c_0", "c_1"]
fA0s = [0.09, 0.03]
observables = ["fA", "fB", "xA", "xB"]

create_tsvs(monomer_As, monomer_Bs, condition_ids, fA0s, observables, base_dir)


# Define where your data lives
EXPERIMENT_BASE_DIR = "/PolyPESTO/experiments/experimental_cpe/data"
EXPERIMENT_ID = "p_000"

# Construct paths and load experiment
exp_paths = ExperimentPaths(base_dir=EXPERIMENT_BASE_DIR, exp_id=EXPERIMENT_ID)
exp = Experiment.load(paths=exp_paths, model=ReversibleCPE)

# Run parameter estimation
result = run_parameter_estimation(
    exp,
    config=dict(
        optimize=dict(n_starts=100, method="Nelder-Mead"),
        profile=dict(method="Nelder-Mead"),
        sample=dict(n_samples=10000, n_chains=5),
    ),
    overwrite=True,
)

# Visualization
axs = plot_all_measurements(exp.petab_problem.measurement_df)
fig = plt.gcf()
fig.savefig(exp.paths.measurements_data_plot, dpi=300)

fig, ax = plot_optimization_scatter(result)
fig.savefig(exp.paths.optimization_scatter_plot, dpi=300)

fig, ax = plot_waterfall(result)
fig.savefig(exp.paths.waterfall_plot, dpi=300)

fig, ax = plot_sampling_scatter(result)
fig.savefig(exp.paths.sampling_scatter_plot, dpi=300)

fig, ax = plot_parameter_traces(result)
fig.savefig(exp.paths.sampling_trace_plot, dpi=300)

fig, ax = plot_profiles(result)
fig.savefig(exp.paths.profile_plot, dpi=300)

fig, ax = plot_confidence_intervals(result)
fig.savefig(exp.paths.confidence_intervals_plot, dpi=300)

plt.close("all")
