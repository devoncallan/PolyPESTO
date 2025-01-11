import os
from typing import Dict, Iterable, Tuple, Sequence, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import petab
import petab.v1.C as C
import amici
from libsbml import Model

from src.utils import amici as am
from src.utils import sbml as sbml
from src.models import cpe_models as CPE


@dataclass
class PetabParameter:
    id: str
    scale: str
    bounds: Tuple[float, float]
    nominal_value: float
    estimate: bool


############################
### Define petab problem ###
############################


def define_parameters(parameters: List[PetabParameter]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                C.PARAMETER_ID: param.id,
                C.PARAMETER_SCALE: param.scale,
                C.LOWER_BOUND: param.bounds[0],
                C.UPPER_BOUND: param.bounds[1],
                C.NOMINAL_VALUE: param.nominal_value,
                C.ESTIMATE: param.estimate,
            }
            for param in parameters
        ]
    ).set_index(C.PARAMETER_ID)


def define_observables(
    observables: Dict[str, str], noise_value: float = 0.0
) -> pd.DataFrame:

    observable_ids = list(observables.keys())
    observable_formulas = list(observables.values())

    return pd.DataFrame(
        data={
            C.OBSERVABLE_ID: [f"obs_{id}" for id in observable_ids],
            C.OBSERVABLE_FORMULA: observable_formulas,
            C.NOISE_FORMULA: [noise_value] * len(observable_ids),
        }
    ).set_index(C.OBSERVABLE_ID)


def define_conditions(init_conditions: Dict[str, Sequence[float]]) -> pd.DataFrame:
    # Ensure all sequences have the same length
    lengths = [len(values) for values in init_conditions.values()]
    assert all(
        length == lengths[0] for length in lengths
    ), "All sequences must have the same length"

    # Create condition IDs
    num_conditions = lengths[0]
    condition_ids = [f"c_{i}" for i in range(num_conditions)]

    # Create the DataFrame
    conditions = pd.DataFrame(init_conditions)
    conditions[C.CONDITION_ID] = condition_ids
    conditions[C.CONDITION_NAME] = condition_ids

    # Set 'CONDITION_ID' as the index
    return conditions.set_index(C.CONDITION_ID)


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
    meas_df = pd.concat(meas_dfs, ignore_index=True)

    return meas_df


def define_measurements_amici(
    amici_model: amici.Model,
    timepoints: Sequence[float],
    conditions_df: pd.DataFrame,
    observables_df: pd.DataFrame,
    obs_sigma: float = 0.00,
    meas_sigma: float = 0.005,
    debug_return_rdatas: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, List[amici.ReturnDataView]]:

    measurement_dfs = []
    rdatas = []

    for cond_id, row in conditions_df.iterrows():
        # Extract conditions for this row as a dictionary
        conditions = row.to_dict()

        # Run the simulation with these conditions
        rdata = am.run_amici_simulation(
            amici_model, timepoints, conditions, sigma=meas_sigma
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


def get_meas_from_cpe_sim(
    cpe_ouput: np.ndarray,
    observables_df: pd.DataFrame,
    cond_id: str = "none",
    obs_sigma: float = 0.00,
) -> pd.DataFrame:
    # Transform CPE_output to measurements_df

    # This should throw an error if observables_df has anything other
    # than 'xA' and 'xB' in the C.FORMULA column
    pass


def define_measurements_cpe(
    cpe_model: CPE.Model,
    timepoints: Sequence[float],
    conditions_df: pd.DataFrame,
    observables_df: pd.DataFrame,
    obs_sigma: float = 0.00,
    meas_sigma: float = 0.005,
    approach: str = "izu",
    **kwargs,
) -> pd.DataFrame:
    # Loop over all conditions in conditions_df

    # Call CPE.run_CPE_sim, pass in args and kwargs

    # Use get_meas_from_cpe_sim to transform output to measurements_df

    # return measurements_df
    pass


# Maybe add a petab file directory
def write_petab_files(
    sbml_model_filepath: str,
    parameters_df: pd.DataFrame,
    observables_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
) -> str:

    model_dir = os.path.dirname(sbml_model_filepath)
    model_filename = os.path.basename(sbml_model_filepath)

    petab.v1.write_condition_df(
        conditions_df, os.path.join(model_dir, "conditions.tsv")
    )
    petab.v1.write_measurement_df(
        measurements_df, os.path.join(model_dir, "measurements.tsv")
    )
    petab.v1.write_observable_df(
        observables_df, os.path.join(model_dir, "observables.tsv")
    )
    petab.v1.write_parameter_df(
        parameters_df, os.path.join(model_dir, "parameters.tsv")
    )

    # Define PEtab configuration
    yaml_config = {
        C.FORMAT_VERSION: 1,
        C.PARAMETER_FILE: "parameters.tsv",
        C.PROBLEMS: [
            {
                C.SBML_FILES: [model_filename],
                C.CONDITION_FILES: ["conditions.tsv"],
                C.MEASUREMENT_FILES: ["measurements.tsv"],
                C.OBSERVABLE_FILES: ["observables.tsv"],
            }
        ],
    }

    yaml_filepath = os.path.join(model_dir, f"petab.yaml")
    petab.v1.yaml.write_yaml(yaml_config, yaml_filepath)

    # validate written PEtab files
    problem = petab.v1.Problem.from_yaml(yaml_filepath)
    petab.v1.lint.lint_problem(problem)

    return yaml_filepath
