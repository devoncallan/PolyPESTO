import os
import shutil
import logging
from typing import Dict, Sequence, Tuple, Optional

import amici.petab_import
import pandas as pd
import amici
from petab.v1.C import OBSERVABLE_FORMULA

from src.utils.sbml import _model_name_from_filepath

DEFAULT_SOLVER_OPTIONS = {
    "setAbsoluteTolerance": 1e-10,
}


def get_amici_model_name_and_output_dir(sbml_model_filepath: str) -> Tuple[str, str]:

    model_name = _model_name_from_filepath(sbml_model_filepath)
    model_output_dir = os.path.join("/PolyPESTO/amici_models/", model_name)

    return model_name, model_output_dir


def compile_amici_model(
    sbml_model_filepath: str, observables_df: pd.DataFrame, verbose=False
):

    # Define the amici output directory
    model_name, model_output_dir = get_amici_model_name_and_output_dir(
        sbml_model_filepath
    )

    # Clear the output directory if it exists
    if os.path.exists(model_output_dir):
        print(f"Cleaning existing directory: {model_output_dir}")
        shutil.rmtree(model_output_dir)

    observables = {
        str(observable_id): {"formula": str(formula)}
        for observable_id, formula in observables_df[OBSERVABLE_FORMULA].items()
    }

    print("Compiling AMICI model from SBML.")

    if verbose:
        verbose = logging.DEBUG

    sbml_importer = amici.SbmlImporter(sbml_model_filepath)
    sbml_importer.sbml2amici(
        model_name,
        model_output_dir,
        verbose=verbose,
        observables=observables,
    )


def load_amici_model(
    sbml_model_filepath: str,
    observables_df: pd.DataFrame,
    force_compile=False,
    verbose=False,
) -> amici.Model:

    if force_compile:
        compile_amici_model(sbml_model_filepath, observables_df, verbose=verbose)

    model_name, model_output_dir = get_amici_model_name_and_output_dir(
        sbml_model_filepath
    )

    model_module = amici.import_model_module(model_name, model_output_dir)
    return model_module.getModel()


def set_model_parameters(
    model: amici.Model, parameters: Dict[str, float]
) -> amici.Model:
    """
    Sets the parameters of an AMICI model.

    Args:
        model (amici.Model): The AMICI model.
        parameters (Dict[str, float]): The parameters to set.

    Returns:
        amici.Model: The AMICI model with parameters set.
    """

    for param_id, value in parameters.items():
        model.setParameterByName(param_id, value)
    return model


def get_solver(model: amici.Model, **solver_options) -> amici.Solver:

    # Check if kwarg are valid solver functions and then call the function to set the value
    solver = model.getSolver()
    for method, value in solver_options.items():
        if not hasattr(solver, method):
            raise ValueError(f"Solver does not have method: {method}")
        solver_func = getattr(solver, method)
        if not callable(solver_func):
            raise ValueError(f"Solver method {method} is not callable")
        solver_func(value)
    return solver

def run_amici_simulation(
    model: amici.Model,
    timepoints: Sequence[float],
    conditions: Dict[str, float],
    sigma: float = 0.0,
    solver: Optional[amici.Solver] = None,
) -> amici.ReturnDataView:

    if solver is None:
        solver = get_solver(model, **DEFAULT_SOLVER_OPTIONS)

    # Get valid state IDs from the model
    state_ids = model.getStateIds()

    # Dynamically set initial states
    init_states = list(model.getInitialStates())
    for i, state_id in enumerate(state_ids):
        if state_id in conditions:
            init_states[i] = conditions[state_id]

    # Set the model's parameters if in the conditions
    for pname in model.getParameterIds():
        if pname in conditions:
            model.setParameterById(pname, conditions[pname])

    model.setInitialStates(init_states)
    model.setTimepoints(timepoints)

    rdata = amici.runAmiciSimulation(model, solver)
    return rdata
