import numpy as np
import matplotlib.pyplot as plt

import pypesto
import pypesto.sample

from polypesto.core.params import ParameterGroup
from polypesto.core.study import Study, create_study
from polypesto.core.experiment import create_simulation_conditions
from polypesto.models.CRP2 import IrreversibleCPE
from polypesto.core.pypesto import (
    create_ensemble,
    predict_with_ensemble,
)
from polypesto.visualization import plot_all_results, plot_all_comparisons_2D, plot_all_ensemble_predictions
from polypesto.utils.paths import setup_data_dirs
from experiments.irreversible_cpe.util import get_test_study

DATA_DIR, TEST_DIR = setup_data_dirs(__file__)

simulation_params = ParameterGroup.create_parameter_grid(
    {
        # "rA": [0.1, 0.5, 1.0, 2.0, 10.0],
        # "rB": [0.1, 0.5, 1.0, 2.0, 10.0],
        "rA": [10.0],
        "rB": [0.5]
    },
    filter_fn=lambda p: p["rA"] >= p["rB"],
)

# Define fitting parameters
fit_params = IrreversibleCPE.get_default_parameters()

# Define experimental configurations
t_eval = np.arange(0, 1, 0.1)

# fA0s = [[0.1, 0.5], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]]
fA0s = [
    [round(i * 0.1, 1), round(j * 0.1, 1)]
    for i in range(1, 10)
    for j in range(1, 10)
    if i != j
]
ntrials = len(fA0s)
cM0s = [[1.0, 1.0] for _ in range(ntrials)]
names = [f"fA0_{fA0}_cM0_{cM0}" for fA0, cM0 in zip(fA0s, cM0s)]
assert len(fA0s) == len(cM0s), "fA0s and cM0s must have the same length"

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
    overwrite=False,
)
# study = Study.load(DATA_DIR, IrreversibleCPE)

# Run parameter estimation
study.run_parameter_estimation(
    config=dict(
        optimize=dict(n_starts=100, method="Nelder-Mead"),
        profile=dict(method="Nelder-Mead"),
        sample=dict(n_samples=50000, n_chains=3),
    ),
    overwrite=False,
)

plot_all_comparisons_2D(study)
plot_all_results(study)
plot_all_ensemble_predictions(study, get_test_study(study, TEST_DIR))
