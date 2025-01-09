import os
import shutil
import logging
from typing import Dict, Sequence, Tuple

import pandas as pd
import amici
from petab.v1.C import OBSERVABLE_FORMULA

from .sbml import _model_name_from_filepath


def compile_amici_model(
    sbml_model_filepath: str, observables_df: pd.DataFrame, verbose=False
) -> Tuple[str, str]:

    # Define the amici output directory
    model_name = _model_name_from_filepath(sbml_model_filepath)
    model_output_dir = os.path.join("/SBML/amici_models/", model_name)

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

    return model_name, model_output_dir


def load_amici_model(model_name: str, model_output_dir: str) -> amici.Model:
    """
    Loads an AMICI model from a compiled model directory."""

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


def run_amici_simulation(
    model: amici.Model,
    timepoints: Sequence[float],
    conditions: Dict[str, float],
    sigma: float = 0.0,
) -> amici.ReturnDataView:

    solver = model.getSolver()
    solver.setAbsoluteTolerance(1e-10)

    # Get valid state IDs from the model
    state_ids = model.getStateIds()

    # Filter conditions to include only those matching state IDs
    valid_conditions = {
        key: value for key, value in conditions.items() if key in state_ids
    }

    # Dynamically set initial states
    init_states = list(model.getInitialStates())
    for i, state_id in enumerate(state_ids):
        if state_id in valid_conditions:
            init_states[i] = valid_conditions[state_id]

    model.setInitialStates(init_states)
    model.setTimepoints(timepoints)

    rdata = amici.runAmiciSimulation(model, solver)

    return rdata
