import numpy as np
import os

from polypesto.core.params import ParameterGroup
from polypesto.core.study import Study, create_study
from polypesto.core.experiment import create_simulation_conditions
from polypesto.models.CRP2 import IrreversibleCPE


# Define the data directory relative to this script
DATA_DIR = os.path.join(os.path.dirname(__file__), "data/study_000")
os.makedirs(DATA_DIR, exist_ok=True)
TEST_DIR = os.path.join(DATA_DIR, "test")
os.makedirs(TEST_DIR, exist_ok=True)

# Create a parameter grid
simulation_params = ParameterGroup.create_parameter_grid(
    {
        "rA": [0.5, 1.0, 2.0],
        "rB": [0.5, 1.0, 2.0],
    }
)

# Define fitting parameters
fit_params = IrreversibleCPE.get_default_parameters()

# Define experimental configurations
ntrials = 2
t_eval = np.arange(0, 1, 0.1)

# Define experimental conditions
fA0s = [[0.25, 0.5], [0.5, 0.75]]
cM0s = [[1.0, 1.0], [1.0, 1.0]]
names = [f"fA0_{fA0}_cM0_{cM0}" for fA0, cM0 in zip(fA0s, cM0s)]

# Create a test study for ensemble
test_study = create_study(
    model=IrreversibleCPE,
    simulation_params=simulation_params,
    conditions=create_simulation_conditions(
        dict(
            names=["Test_Study"],
            t_eval=[t_eval] * 27,
            conditions=dict(
                fA0=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 3],
                cM0=[[0.5] * 9, [1.0] * 9, [2.0] * 9],
                fit_params=[fit_params] * 27,
                noise_level=[0.02] * 27,
            ),
        )
    ),
    base_dir=DATA_DIR,
    overwrite=False,
)

conditions = create_simulation_conditions(
    dict(
        name=names,
        t_eval=[t_eval] * ntrials,
        conditions=dict(fA0=fA0s, cM0=cM0s),
        fit_params=[fit_params] * ntrials,
        noise_level=[0.02] * ntrials,
    )
)

# Create the study - this will simulate all experiments
study = create_study(
    model=IrreversibleCPE,
    simulation_params=simulation_params,
    conditions=conditions,
    base_dir=DATA_DIR,
    overwrite=True,
)

study = Study.load(DATA_DIR, IrreversibleCPE)

# # Run parameter estimation
study.run_parameter_estimation(
    config=dict(
        optimize=dict(n_starts=10, method="Nelder-Mead"),
        profile=dict(method="Nelder-Mead"),
        sample=dict(n_samples=10000, n_chains=3),
    )
)
import matplotlib.pyplot as plt

for (cond_id, p_id), result in study.results.items():

    exp = study.experiments[(cond_id, p_id)]
    true_params = exp.true_params

    print(f"Experiment: {cond_id}, Parameter set: {p_id}")
    print(f"  Result: {result}")

    result.summary(full=True)

    from polypesto.visualization.plots import (
        plot_sampling_scatter,
        plot_confidence_intervals,
        plot_waterfall,
        plot_parameter_traces,
    )

    fig, ax = plot_waterfall(result)
    fig.savefig(exp.paths.waterfall_plot, dpi=300)

    fig, ax = plot_sampling_scatter(result, true_params.to_dict())
    fig.savefig(exp.paths.sampling_scatter_plot, dpi=300)

    fig, ax = plot_parameter_traces(result, true_params.to_dict())
    fig.savefig(exp.paths.sampling_trace_plot, dpi=300)

    # fig, ax = plot_confidence_intervals(result, true_params.to_dict())
    # fig.savefig(exp.paths.confidence_intervals_plot, dpi=300)
