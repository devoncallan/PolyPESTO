"""
Copolymer Equation Parameter Estimation with Single Reaction Experiment.

This module provides functions to configure and run parameter estimation experiments
for the irreversible copolymerization model.

HYPOTHESIS:
- Parameter identifiability will vary with feed fraction conditions
- Low feed fractions will be more challenging for accurate parameter estimation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import shutil
import os

import pypesto
from polypesto.models.CRP2 import IrreversibleCPE as Model
from polypesto.core.params import ParameterGroup, ParameterSet
import polypesto.core.petab as pet
from polypesto.core.pypesto import (
    create_problem_set,
    run_parameter_estimation,
    load_pypesto_problem,
)
from polypesto.core.experiments import load_all_experiments

# Data directory for experimental data and results
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Default parameter estimation configuration
PE_CONFIG = {
    'optimize': {
        'n_starts': 100, 
        'method': 'Nelder-Mead'
    },
    'profile': {
        'method': 'Nelder-Mead'
    },
    'sample': {
        'n_samples': 10000,
        'n_chains': 3
    }
}

def define_parameters() -> ParameterGroup:
    """
    Define parameter sets for reactivity ratios to be tested.
    
    Creates a grid of parameter combinations for rA and rB values.
    
    Returns
    -------
    ParameterGroup
        Group containing all parameter sets to test
    """
    # Reactivity ratio values to test
    rA = [0.1, 0.5, 1.0, 2.0, 10.0]
    rB = [0.1, 0.5, 1.0, 2.0, 10.0]

    # Create parameter group with all combinations
    pg = ParameterGroup(Model.name, {})
    for _rA in rA:
        for _rB in rB:
            pg.lazy_add({"rA": _rA, "rB": _rB})

    return pg

def define_experiment(
    t_eval: np.ndarray, 
    feed_fractions: np.ndarray, 
    concentrations: np.ndarray
) -> Tuple[str, pet.PetabData]:
    """
    Define experimental conditions and observables for a specific feed fraction.
    
    Parameters
    ----------
    t_eval : np.ndarray
        Time points for evaluation
    feed_fractions : np.ndarray
        Feed fraction values for experimental conditions
    concentrations : np.ndarray
        Monomer concentrations for experimental conditions
    
    Returns
    -------
    Tuple[str, pet.PetabData]
        Directory path and PEtab data object
    """
    # Create directory name based on feed fraction
    dir_path = os.path.join(DATA_DIR, f"fA0_{feed_fractions[0]:.2f}")

    # Define fitting parameters
    params_dict = Model.get_default_fit_params()
    param_df = pet.define_parameters(params_dict)

    # Define experimental conditions
    cond_df = Model.create_conditions(feed_fractions, concentrations)
    obs_df = Model.get_default_observables()
    empty_meas_df = pet.define_empty_measurements(obs_df, cond_df, t_eval)

    # Create and return PEtab data
    petab_data = pet.PetabData(
        obs_df=obs_df, 
        cond_df=cond_df, 
        param_df=param_df, 
        meas_df=empty_meas_df
    )
    
    return dir_path, petab_data

def generate_experiment_data() -> Tuple[List[ParameterGroup], List[PetabPaths]]:
    """
    Generate experiment data for all feed fraction conditions.
    
    This function:
    1. Sets up experiment conditions for different feed fractions
    2. Creates parameter sets to test
    3. Generates synthetic data and PEtab files
    
    Returns
    -------
    Tuple[List[ParameterGroup], List[PetabPaths]]
        Parameter groups and PEtab paths for all experiments
    """
    # Create data directory if it doesn't exist (or clean it if it does)
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Define experiment conditions
    t_eval = np.arange(0, 1, 0.1, dtype=float)
    feed_fractions = np.array([[0.1], [0.25], [0.5], [0.75]])
    concentrations = np.array([[1.0], [1.0], [1.0], [1.0]])

    petab_paths = []
    param_groups = []
    
    # Generate data for each feed fraction
    for fA0, cM0 in zip(feed_fractions, concentrations):
        dir_path, petab_data = define_experiment(t_eval, fA0, cM0)
        param_group = define_parameters()
        
        # Create problem set with model, parameters and data
        paths = create_problem_set(Model, param_group, petab_data, dir_path)

        param_groups.append(param_group)
        petab_paths.append(paths)
    
    return param_groups, petab_paths

def run_parameter_estimation(
    config: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Dict[str, pypesto.Result]]:
    """
    Run parameter estimation for all experiments.
    
    Loads all experiments from the data directory and runs parameter estimation
    for each parameter set in each experiment.
    
    Parameters
    ----------
    config : Dict[str, Dict[str, Any]], optional
        Configuration options for parameter estimation steps.
        Can contain keys 'optimize', 'profile', 'sample' with 
        respective options. If None, the default config will be used.
        
    Returns
    -------
    Dict[str, Dict[str, pypesto.Result]]
        Results dictionary mapping experiment names to parameter set results
    """
    # Use default config if none provided
    if config is None:
        config = PE_CONFIG
    
    # Load all experiments
    experiments = load_all_experiments(DATA_DIR, Model.name)
    
    # Run parameter estimation
    results = run_parameter_estimation(experiments, config)
    
    return results

"""
// True parameters to use for estimation
true_params = create_parameter_grid(
    param_ranges = {
        "rA": [0.1, 0.5, 1.0, 2.0, 10.0],
        "rB": [0.1, 0.5, 1.0, 2.0, 10.0]
    },
    group_id = "parameter_grid"
)

// Needs to match the arguments in model interface
conditions = {
    "fA0s": [[0.1], [0.25], [0.5], [0.75]],
    "cM0s": [[1.0], [1.0], [1.0], [1.0]]
}
time_points = np.arange(0, 1, 0.1)

// Defining parameter dataframes
// Predefined parameter datafames (setting estimate to True, etc.)
fit_params = [MODEL_0000, MODEL_0001, MODEL_0010, MODEL_0011]

// Create a study object
num_trials = len(fit_params) = len(conditions["fA0s"]) = ...

study = Study.create(
    model=Model,
    true_params=true_params,
    fit_params=fit_params,
    conditions=conditions,
    time_points=[time_points] * num_trials,
    noise=[0.02] * num_trials,
)
// Collection of all experiments (i.e., for all conditions for all sets of parameters)

// Generate data and write to PEtab files
study.run(data_dir=DATA_DIR, overwrite=False)

// Parameter estimation and sampling, etc.
config = {
    'optimize': {'n_starts': 200, 'method': 'scipy-lbfgsb'},
    'profile': {'ratio_bound': 10},
    'sample': {'n_samples': 5000, 'n_chains': 4}
}
estimator = ParameterEstimator(model=Model)
results = estimator.estimate_parameters(study.experiments, config=config)


// 
"""
