from typing import Dict
import os

from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df
from pypesto.petab import PetabImporter

from polypesto.utils.paths import PetabPaths
from polypesto.core.petab import PetabData, PetabIO
from polypesto.core.params import ParameterGroup, ParameterSet
from polypesto.models import sbml


def load_pypesto_problem(yaml_path: str, model_name: str, **kwargs):

    importer = PetabImporter.from_yaml(
        yaml_path,
        model_name=model_name,
        base_path="",
    )
    problem = importer.create_problem(**kwargs)

    return importer, problem

#########################
### Write PETab files ###
#########################


def write_initial_petab(
    model_def: sbml.ModelDefinition,
    pg: ParameterGroup,
    data: PetabData,
    model_dir: str,
) -> PetabPaths:

    model_name = os.path.basename(model_dir)
    exp_name = str(data.name)
    data_dir = os.path.join(model_dir, "data", exp_name)
    paths = PetabPaths(data_dir)
    print(paths.base_dir)

    sbml_filepath = sbml.write_model(
        model_def=model_def, model_filepath=paths.model(model_name)
    )

    PetabIO.write_obs_df(data.obs_df, filename=paths.observables)
    PetabIO.write_cond_df(data.cond_df, filename=paths.conditions)
    PetabIO.write_param_df(data.param_df, filename=paths.fit_parameters)

    for p_id in pg.get_ids():
        paths.make_exp_dir(p_id)

        # Write the true parameters to file
        pg.by_id(p_id).write(paths.params(p_id))

        PetabIO.write_meas_df(data.meas_df, filename=paths.measurements(p_id))
        PetabIO.write_yaml(
            yaml_filepath=str(paths.petab_yaml(p_id)),
            sbml_filepath=sbml_filepath,
            cond_filepath=paths.conditions,
            meas_filepath=paths.measurements(p_id),
            obs_filepath=paths.observables,
            param_filepath=paths.fit_parameters,
        )

    return paths


def create_problem_set(
    model_def: sbml.ModelDefinition,
    pg: ParameterGroup,
    data: PetabData,
    force_compile=False,
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
    
    :return:
        Dictionary of YAML paths for each parameter set in ``pg``.
    """

    model_name = str(model_def.__name__)
    model_dir = f"/PolyPESTO/experiments/{model_name}"

    # Write without simulated data first
    paths = write_initial_petab(model_def, pg, data, model_dir=model_dir)

    yaml_paths = paths.find_yaml_paths()
    yaml_path = list(yaml_paths.values())[0]

    importer, problem = load_pypesto_problem(
        yaml_path, str(model_name), force_compile=force_compile
    )

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

        PetabIO.write_meas_df(meas_df, filename=paths.measurements(p_id))

    return model_name, yaml_paths
