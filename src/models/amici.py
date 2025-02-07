import os
import shutil
import logging
from typing import Dict, Sequence, Optional

import numpy as np
import pandas as pd
import amici
import petab.v1.C as C

from src.utils import sbml
from src.utils.params import ParameterSet
import src.utils.file as file
import src.utils.petab as pet
from src.models.model import Model

DEFAULT_SOLVER_OPTIONS = {
    "setAbsoluteTolerance": 1e-10,
}

AMICI_MODELS_DIR = "/PolyPESTO/.amici_models/"


def compile_amici_model(
    name: str,
    output_dir: str,
    sbml_filepath: str,
    observables_df: pd.DataFrame,
    verbose=False,
):

    # Clear the output directory if it exists
    # Replace with file module function
    if os.path.exists(output_dir):
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    observables = {
        str(observable_id): {"formula": str(formula)}
        for observable_id, formula in observables_df[C.OBSERVABLE_FORMULA].items()
    }

    print("Compiling AMICI model from SBML.")

    if verbose:
        verbose = logging.DEBUG

    sbml_importer = amici.SbmlImporter(sbml_filepath)
    sbml_importer.sbml2amici(
        name,
        output_dir,
        verbose=verbose,
        observables=observables,
    )


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
    t_eval: Sequence[float],
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
    model.setTimepoints(t_eval)

    rdata = amici.runAmiciSimulation(model, solver)
    return rdata


def get_meas_from_amici_sim(
    rdata: amici.ReturnDataView,
    observables_df: pd.DataFrame,
    cond_id: str = "none",
    obs_sigma: float = 0.00,
) -> pd.DataFrame:

    meas_dfs = []
    for obs_id in observables_df.index:

        obs_data = rdata.by_id(obs_id)
        obs_data = np.array(obs_data) * (1 + obs_sigma * np.random.randn(len(obs_data)))
        num_pts = len(obs_data)

        obs_meas_df = pd.DataFrame(
            {
                C.OBSERVABLE_ID: [obs_id] * num_pts,
                C.SIMULATION_CONDITION_ID: [cond_id] * num_pts,
                C.TIME: rdata.ts,
                C.MEASUREMENT: obs_data,
            }
        )
        meas_dfs.append(obs_meas_df)
    meas_df = pd.concat(meas_dfs)

    meas_df = pet.PetabIO.format_meas_df(meas_df)
    return meas_df


class AmiciModel(Model):

    def __init__(
        self,
        name: str,
        model: amici.Model,
        obs_df: pd.DataFrame,
        model_filepath: file.Filepath,
        **kwargs,
    ):
        super().__init__(
            name=name, model=model, obs_df=obs_df, model_filepath=model_filepath
        )

    def simulate(
        self,
        t_eval: Sequence[float],
        conditions: Dict[str, float],
        cond_id: str = None,
        **kwargs,
    ) -> pd.DataFrame:

        rdata = run_amici_simulation(self.model, t_eval, conditions, **kwargs)
        return get_meas_from_amici_sim(rdata, self.obs_df, cond_id=cond_id)

    def set_params(self, parameters: ParameterSet):
        for param in parameters.parameters.values():
            self.model.setParameterByName(param.id, param.value)


def create_model(
    model_def: sbml.ModelDefinition,
    obs_df: pd.DataFrame,
    model_dir: Optional[file.Directory] = None,
    force_compile: bool = False,
    verbose: bool = False,
) -> AmiciModel:
    """
    Model dir is the name of the SBML model output directory"""

    name = model_def.__name__

    sbml_filepath = sbml.write_model(name, model_def, model_dir)

    amici_dir = os.path.join(AMICI_MODELS_DIR, name)

    if force_compile:
        compile_amici_model(name, amici_dir, sbml_filepath, obs_df, verbose=verbose)

    model_module = amici.import_model_module(name, amici_dir)
    model = model_module.getModel()

    return AmiciModel(
        name=name, model=model, obs_df=obs_df, model_filepath=sbml_filepath
    )
