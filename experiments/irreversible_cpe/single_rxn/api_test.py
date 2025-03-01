import numpy as np
import os

from polypesto.core.params import ParameterGroup
from polypesto.core.study import Study, create_study
from polypesto.core.experiment import ExperimentConfig, create_experiment_configs
from polypesto.models.CRP2 import IrreversibleCPE


# Define the data directory relative to this script
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Create a small parameter grid for testing
true_params = ParameterGroup.create_parameter_grid(
    {
        "rA": [0.5, 1.0, 2.0],
        "rB": [0.5, 1.0, 2.0],
    }
)

# Define experimental configurations
ntrials = 2
t_eval = np.arange(0, 1, 0.1)

# Simplified arrays - single values instead of nested lists
fA0s = [[0.5], [0.25]]
cM0s = [[1.0], [1.0]]
names = [f"fA0_{fA0[0]}_cM0_{cM0[0]}" for fA0, cM0 in zip(fA0s, cM0s)]
fit_params = IrreversibleCPE.get_default_parameters()

# Create experiment configs dictionary
exp_configs_dict = dict(
    name=names,
    t_eval=[t_eval] * ntrials,
    conditions=dict(fA0=fA0s, cM0=cM0s),
    fit_params=[fit_params] * ntrials,
    noise_level=[0.02] * ntrials,
)

# Convert to list of ExperimentConfig objects
exp_configs = create_experiment_configs(exp_configs_dict)

# # Create the study - this will simulate all experiments
# print("Creating study with multiple experiments...")
study = create_study(
    model=IrreversibleCPE, 
    true_params=true_params, 
    configs=exp_configs,
    base_dir=DATA_DIR,
    overwrite=True,
)

study.run_parameter_estimation()

# # Print summary of created study
# print(f"\nCreated study with {study.num_trials} trials")
# print(f"Total experiments: {len(study.experiments)}")
# print(f"Trial names: {study.trial_names}")
# print(f"Parameter sets: {len(study.true_params.get_ids())}")

# # Test saving and loading the study
# print("\nTesting save functionality...")
# study.save()  # Save to the existing directory

print("\nTesting load functionality...")
loaded_study = Study.load(DATA_DIR, IrreversibleCPE)

# Verify the loaded study matches the original
print(f"Loaded study has {loaded_study.num_trials} trials")
print(f"Loaded study has {len(loaded_study.experiments)} experiments")
print(f"Loaded study has parameter sets: {len(loaded_study.true_params.get_ids())}")

# Test convenience methods
print("\nTesting convenience methods:")
# Get experiment by trial and parameter set
trial_id = loaded_study.trial_names[0]
p_id = loaded_study.true_params.get_ids()[0]
experiment = loaded_study.get_experiment(trial_id, p_id)
print(f"Retrieved experiment for trial {trial_id} and parameter set {p_id}")

# Get all experiments for a trial
exps_by_trial = loaded_study.get_experiments_by_trial(trial_id)
print(f"Found {len(exps_by_trial)} experiments for trial {trial_id}")

# Get all experiments for a parameter set
exps_by_param = loaded_study.get_experiments_by_parameter(p_id)
print(f"Found {len(exps_by_param)} experiments for parameter set {p_id}")
