"""
MOTIVATION:
- Understand how well fitting works with a single feed fraction experiment. 

HYPOTHESIS:
- Low feed fractions will be difficult to fit. 

"""
import numpy as np

from polypesto.models.CRP2 import IrreversibleCPE as Model
from polypesto.core.params import ParameterGroup
import polypesto.core.petab as pet
from polypesto.core.pypesto import create_problem_set, load_pypesto_problem
from polypesto.core.params import ParameterSet, ParameterGroup

## Cleanly define true parameter sets
def parameters() -> ParameterGroup:
    rA = [0.1, 0.5, 1.0, 2.0, 10.0]
    rB = [0.1, 0.5, 1.0, 2.0, 10.0]
    
    pg = ParameterGroup()
    for rA in rA:
        for rB in rB:
            pg.add(ParameterSet.from_dict({"rA": rA, "rB": rB}))

    return pg

def experiment(t_eval, fA0s, cM0s) -> pet.PetabData:

    # Define fitting parameters 
    params_dict = Model.get_default_fit_params()
    param_df = pet.define_parameters(params_dict)

    # Define experimental conditions
    cond_df = Model.create_conditions(fA0s, cM0s)
    obs_df = Model.get_default_observables()
    empty_meas_df = pet.define_empty_measurements(obs_df, cond_df, t_eval)

    return pet.PetabData(
        obs_df=obs_df, cond_df=cond_df, param_df=param_df, meas_df=empty_meas_df
    )

# Define constants
t_eval = np.arange(0, 1, 0.1, dtype=float)
fA0s = [[0.1], [0.25], [0.5]]
cM0s = [[1.0], [1.0], [1.0]]

for fA0, cM0 in zip(fA0s, cM0s):
    
    data = experiment(t_eval, fA0, cM0)
    paths = create_problem_set(Model.create_sbml_model(), parameters(), data)
    
    for path in paths:
        importer, problem = load_pypesto_problem(path, Model.name)
        
        vis.plot_measurements(problem, data, path)
        
        result = optimize(problem)
        
        result = sample(problem, result, n_samples=1000)
        
        vis.plot_samples(problem, result, path)
        vis.plot_parameter_estimates(problem, result, path)
        
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
