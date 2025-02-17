from typing import Dict
import os

import pypesto
from pypesto.petab import PetabImporter
import src.utils.petab as pet

from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df
from src.utils.params import ParameterGroup, ParameterSet
from src.utils.paths import PetabPaths


import src.utils.sbml as sbml
import pandas as pd


import numpy as np

AMICI_MODELS_DIR = "/PolyPESTO/amici_models/"
FITTING_DIR = "/PolyPESTO/src/data/fitting/"

def load_pypesto_problem(yaml_path: str, model_name: str, **kwargs):

    importer = PetabImporter.from_yaml(
        yaml_path,
        # output_folder=AMICI_MODELS_DIR,
        model_name=model_name,
        base_path="",
    )
    problem = importer.create_problem(**kwargs)

    print("Problem created!")
    petab_problem = importer.petab_problem
    amici_model = problem.objective.amici_model
    amici_solver = problem.objective.amici_solver

    return importer, problem


def create_problem_set(
    model_def: sbml.ModelDefinition,
    pg: ParameterGroup,
    data: pet.PetabData,
    force_compile=False,
    data_dir: str = FITTING_DIR
) -> Dict[str, str]:
    """Create Petab problem set by simulating data.
    
    :param model_def:
        SBML Model definition.
    :param pg:
        Parameter group (multiple sets of parameters) to generate data.
    :param data:
        Contains observables, conditions, measurements, fit params.
    :param force_compile:
        Force recompilation of model.
    :param data_dir:
        Directory to save data.
        
    :return:
        Dictionary of YAML paths for each parameter set in ``pg``.
    """
    
    data_dir = os.path.join(data_dir, model_def.__name__)

    # Write without simulated data first
    paths = pet.write_initial_petab(data_dir, model_def, pg, data)

    yaml_paths = paths.find_yaml_paths()
    yaml_path = list(yaml_paths.values())[0]

    importer, problem = load_pypesto_problem(
        yaml_path, str(model_def.__name__), force_compile=force_compile
    )
    print(str(model_def.__name__) + " loaded!")

    for p_id, yaml_path in yaml_paths.items():

        params_path = paths.params(p_id)
        params = ParameterSet.load(params_path).to_dict()

        sim_data = simulate_petab(
            petab_problem=importer.petab_problem,
            amici_model=problem.objective.amici_model,
            solver=problem.objective.amici_solver,
            problem_parameters=params,
        )
        meas_df = rdatas_to_measurement_df(
            sim_data["rdatas"],
            problem.objective.amici_model,
            importer.petab_problem.measurement_df,
        )

        pet.PetabIO.write_meas_df(meas_df, filename=paths.measurements(p_id))

    return yaml_paths
