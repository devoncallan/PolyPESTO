"""
MOTIVATION:
- Understand how well fitting works with a single feed fraction experiment. 

HYPOTHESIS:
- Low feed fractions will be difficult to fit. 

"""

import numpy as np
from typing import Tuple

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

from polypesto.models.equilibrium import EquilibriumODE as Model
from polypesto.core.params import ParameterGroup
import polypesto.core.petab as pet
from polypesto.core.pypesto import create_problem_set, load_pypesto_problem
from polypesto.core.params import ParameterSet, ParameterGroup
from polypesto.utils.plot import plot_all_measurements

from pathlib import Path

import os

DIR_NAME = os.path.basename(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ************** Define parameters **************
def parameters() -> ParameterGroup:

    k1 = [0.1]
    k2 = [0.5]

    pg = ParameterGroup(DIR_NAME, {})
    for _k1 in k1:
        for _k2 in k2:
            pg.lazy_add({"k1": _k1, "k2": _k2})

    return pg


# ************** Define experiments **************
# experimental conditions, observables, and fit parameters
def experiment(t_eval, fA0s, cM0s) -> Tuple[str, pet.PetabData]:

    dir = os.path.join(DATA_DIR, f"fA0_{fA0s[0]:.2f}")

    # Define fitting parameters
    params_dict = Model.get_default_fit_params()
    param_df = pet.define_parameters(params_dict)

    # Define experimental conditions
    cond_df = Model.create_conditions(fA0s, cM0s)
    obs_df = Model.get_default_observables()
    empty_meas_df = pet.define_empty_measurements(obs_df, cond_df, t_eval)

    return dir, pet.PetabData(
        obs_df=obs_df, cond_df=cond_df, param_df=param_df, meas_df=empty_meas_df
    )


# ************** Define experiments **************
t_eval = np.arange(0, 1, 0.1, dtype=float)
fA0s = np.array([[0.1], [0.25], [0.5]])
cM0s = np.array([[1.0], [1.0], [1.0]])

for fA0, cM0 in zip(fA0s, cM0s):

    print(fA0, cM0)
    dir, data = experiment(t_eval, fA0, cM0)
    yaml_paths = create_problem_set(Model, parameters(), data, dir)
    # print(paths)

    yaml_path = yaml_paths["p_000"]
    importer, problem = load_pypesto_problem(yaml_path=yaml_path, model_name=Model.name)

    plot_all_measurements(
        importer.petab_problem.measurement_df,
        # group_by=C.SIMULATION_CONDITION_ID,
        group_by=pet.C.OBSERVABLE_ID,
        format_axes_kwargs={
            "set_xlabel": "Total Conversion",
            "set_ylabel": "Monomer Conversion",
            # "set_xlim": (0, 1),
            # "set_ylim": (0, 1),
        },
        plot_style="both",
        alpha=0.5,
    )
    plt.show()
    
    
    
    break

    # for name, path in yaml_paths.items():
    #     print("Path:", path)
    #     importer, problem = load_pypesto_problem(path, Model.name)

    #     vis.plot_measurements(problem, data, path)

    #     result = optimize(problem)

    #     result = sample(problem, result, n_samples=1000)

    #     vis.plot_samples(problem, result, path)
    #     vis.plot_parameter_estimates(problem, result, path)

    pass
"""
Experiment takes in parameter group and a petab data object.
- Creates petab problem for each set of parameters

for fA0, cM0 in zip(fA0s, cM0s):
    data = experiment(t_eval, fA0, cM0)
    paths = create_problem_set(Model.create_sbml_model(), parameters(), experiment(t_eval, fA0, cM0))
    
    for path in paths: # Looping through each parameter set
        importer, problem = load_pypesto_problem(yaml_path=path, Model)
        result = optimize(problem)
        sample(problem, result, n_samples=1000)
        
        Save sampling results, save figures, etc.
        

"""
# exp = Experiment(standard_library, fA0s=fA0, cM0s=cM0, t_eval=t_eval)

# Sets up the fitting routine for given conditions


# exp.fit(solver_params, )

# exp.fit(overwrite=False, save_figs=True) # A lot happens here under the hood.

# How do we then analyze this? Jupyter notebook?
