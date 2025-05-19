import numpy as np

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup
from polypesto.core.study import Study, create_study
from polypesto.core.experiment import create_simulation_conditions
from polypesto.models.CRP2 import ReversibleCPE
from polypesto.utils.paths import setup_data_dirs
from polypesto.visualization import (
    plot_all_results,
    plot_all_ensemble_predictions,
    plot_all_comparisons_1D,
)


from experiments.irreversible_cpe.util import get_test_study

# from .util import get_test_study

DATA_DIR, TEST_DIR = setup_data_dirs(__file__)

simulation_params = ParameterGroup.create_parameter_grid(
    {"rA": [10.0], "rB": [0.5], "KAA": [0.2]},
    filter_fn=lambda p: p["rA"] >= p["rB"],
)

# Define fitting parameters
fit_params = ReversibleCPE.get_default_fit_params()
fit_params["KAA"].estimate = True
fit_params = pet.define_parameters(fit_params)

obs_df = ReversibleCPE.create_observables(
    observables={"fA": "fA", "fB": "fB", "xA": "xA", "xB": "xB"}, noise_value=0.02
)


# Define experimental configurations
t_eval = np.arange(0, 0.6, 0.01)
fA0s = [[0.3]]
cM0s = [[1.0]]
# fA0s = [[0.3], [0.5], [0.7]]
# cM0s = [[1.0], [1.0], [1.0]]
# fA0s = [[0.1], [0.3], [0.5], [0.7], [0.9]]
# cM0s = [[1.0], [1.0], [1.0], [1.0], [1.0]]
names = [f"fA0_{fA0}_cM0_{cM0}" for fA0, cM0 in zip(fA0s, cM0s)]
ntrials = len(fA0s)
assert len(fA0s) == len(cM0s), "fA0s and cM0s must have the same length"

conditions = create_simulation_conditions(
    dict(
        name=names,
        t_eval=[t_eval] * ntrials,
        conditions=dict(fA0=fA0s, cM0=cM0s, ),
        fit_params=[fit_params] * ntrials,
        noise_level=[0.005] * ntrials,
    )
)
# Create the study - this will simulate all experiments
study = create_study(
    model=ReversibleCPE,
    simulation_params=simulation_params,
    conditions=conditions,
    obs_df=obs_df,
    base_dir=DATA_DIR,
    overwrite=True,
)
print(study)
# Run parameter estimation
study.run_parameter_estimation(
    config=dict(
        optimize=dict(n_starts=100, method="Nelder-Mead"),
        profile=dict(method="Nelder-Mead"),
        sample=dict(n_samples=100000, n_chains=3),
    ),
    overwrite=True,
)
