import os
import shutil
import logging
from typing import Dict, Sequence, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import amici
import petab.v1.C as C

from src.utils import sbml
from src.utils.params import Parameter
import src.utils.file as file
import src.utils.petab as pet

DEFAULT_SOLVER_OPTIONS = {
    "setAbsoluteTolerance": 1e-10,
}


def get_amici_model_name_and_output_dir(sbml_model_filepath: str) -> Tuple[str, str]:

    model_name = sbml._model_name_from_filepath(sbml_model_filepath)
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
        for observable_id, formula in observables_df[C.OBSERVABLE_FORMULA].items()
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


def load_amici_model_from_definition(
    model_fun: sbml.ModelDefinition,
    observables_df: pd.DataFrame,
    model_dir: str,
    **kwargs,
) -> Tuple[str, amici.Model]:

    sbml_model_filepath = sbml.write_model(model_fun=model_fun, model_dir=model_dir)

    validator = sbml.validateSBML(ucheck=False)
    validator.validate(sbml_model_filepath)

    model = load_amici_model(
        sbml_model_filepath, observables_df=observables_df, **kwargs
    )

    return sbml_model_filepath, model


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
    model: amici.Model, parameters: List[Parameter]
) -> amici.Model:
    """
    Sets the parameters of an AMICI model.

    Args:
        model (amici.Model): The AMICI model.
        parameters (List[Parameter]): The parameters to set.

    Returns:
        amici.Model: The AMICI model with parameters set.
    """

    for param in parameters:
        model.setParameterByName(param.id, param.value)
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


def define_measurements_amici(
    model: amici.Model,
    t_eval: Sequence[float],
    conditions_df: pd.DataFrame,
    observables_df: pd.DataFrame,
    obs_sigma: float = 0.00,
    meas_sigma: float = 0.005,
    solver: Optional[amici.Solver] = None,
    debug_return_rdatas: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, List[amici.ReturnDataView]]:

    measurement_dfs = []
    rdatas = []

    for cond_id, row in conditions_df.iterrows():
        # Extract conditions for this row as a dictionary
        conditions = row.to_dict()

        # Run the simulation with these conditions
        rdata = run_amici_simulation(
            model, t_eval, conditions, sigma=meas_sigma, solver=solver
        )
        rdatas.append(rdata)

        # Generate measurements from the simulation
        meas_df = get_meas_from_amici_sim(
            rdata, observables_df, cond_id=str(cond_id), obs_sigma=obs_sigma
        )
        measurement_dfs.append(meas_df)

    measurement_df = pd.concat(measurement_dfs, ignore_index=True)

    if debug_return_rdatas:
        return measurement_df, rdatas
    return measurement_df


from src.models.model import Model


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

        # cond_dict = cond_df.loc[conditions].to_dict(orient="records")[0]
        rdata = run_amici_simulation(self.model, t_eval, conditions, **kwargs)
        return get_meas_from_amici_sim(rdata, self.obs_df, cond_id=cond_id)

    def set_params(self, parameters: List[Parameter]):
        self.model = set_model_parameters(self.model, parameters)


def create_amici_model(
    sbml_model_func: sbml.ModelDefinition,
    obs_df: pd.DataFrame,
    model_dir: Optional[file.Directory] = None,
    **kwargs,
) -> AmiciModel:

    name = sbml_model_func.__name__

    model_filepath, model = load_amici_model_from_definition(
        model_fun=sbml_model_func,
        observables_df=obs_df,
        model_dir=model_dir,
        **kwargs,
    )

    return AmiciModel(
        name=name, model=model, obs_df=obs_df, model_filepath=model_filepath
    )
