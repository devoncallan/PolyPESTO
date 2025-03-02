

# Run pypesto experiment

from experiments.irreversible_cpe.single_rxn.experiments import load_all_experiments
from polypesto.core.pypesto import run_parameter_estimation
from . import exp

DATA_DIR = exp.DATA_DIR

true_param_groups, all_petab_paths = exp.generate_experiment_data()

experiments = load_all_experiments(exp.DATA_DIR, exp.Model.name)

results = run_parameter_estimation(experiments, exp.PE_CONFIG)


