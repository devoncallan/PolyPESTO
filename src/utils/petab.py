import os
from typing import Dict, Iterable, Tuple, Sequence, List, Optional
from dataclasses import dataclass

import pandas as pd
import petab
import petab.v1.C as C


@dataclass
class FitParameter:
    id: str
    scale: str
    bounds: Tuple[float, float]
    nominal_value: float
    estimate: bool


############################
### Define petab problem ###
############################


def define_parameters(params_dict: Dict[str, FitParameter]) -> pd.DataFrame:
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
            for param in params_dict.values()
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


def write_cond_df(
    cond_df: pd.DataFrame, dir: str = None, filename: str = "conditions.tsv"
):

    cond_filepath = os.path.join(dir, filename) if dir else filename
    petab.v1.write_condition_df(cond_df, cond_filepath)
    return cond_filepath


def write_meas_df(
    meas_df: pd.DataFrame, dir: str = None, filename: str = "measurements.tsv"
):

    meas_filepath = os.path.join(dir, filename) if dir else filename
    petab.v1.write_measurement_df(meas_df, meas_filepath)
    return meas_filepath


def write_obs_df(
    obs_df: pd.DataFrame, dir: str = None, filename: str = "observables.tsv"
):

    obs_filepath = os.path.join(dir, filename) if dir else filename
    petab.v1.write_observable_df(obs_df, obs_filepath)
    return obs_filepath


def write_param_df(
    param_df: pd.DataFrame, dir: str = None, filename: str = "parameters.tsv"
):

    param_filepath = os.path.join(dir, filename) if dir else filename
    petab.v1.write_parameter_df(param_df, param_filepath)
    return param_filepath


def write_yaml_file(
    yaml_dir: str,
    sbml_filepath: str = None,
    cond_filepath: str = None,
    meas_filepath: str = None,
    obs_filepath: str = None,
    param_filepath: str = None,
) -> str:

    # Define the PEtab directory
    os.makedirs(yaml_dir, exist_ok=True)
    yaml_filepath = os.path.join(yaml_dir, "petab.yaml")

    yaml_config = {
        C.FORMAT_VERSION: 1,
        C.PARAMETER_FILE: param_filepath,
        C.PROBLEMS: [
            {
                C.SBML_FILES: [sbml_filepath],
                C.CONDITION_FILES: [cond_filepath],
                C.MEASUREMENT_FILES: [meas_filepath],
                C.OBSERVABLE_FILES: [obs_filepath],
            }
        ],
    }
    petab.v1.yaml.write_yaml(yaml_config, yaml_filepath)

    # validate written PEtab files
    problem = petab.v1.Problem.from_yaml(yaml_filepath)
    petab.v1.lint.lint_problem(problem)

    return yaml_filepath


def write_petab_files(
    yaml_dir: str,
    sbml_filepath: str,
    param_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    meas_df: pd.DataFrame,
    param_dir: Optional[str] = None,
    obs_dir: Optional[str] = None,
    cond_dir: Optional[str] = None,
    meas_dir: Optional[str] = None,
) -> str:

    obs_filepath = write_obs_df(obs_df, obs_dir)
    cond_filepath = write_cond_df(cond_df, cond_dir)
    meas_filepath = write_meas_df(meas_df, meas_dir)
    param_filepath = write_param_df(param_df, param_dir)

    yaml_filepath = write_yaml_file(
        yaml_dir,
        sbml_filepath,
        cond_filepath=cond_filepath,
        meas_filepath=meas_filepath,
        obs_filepath=obs_filepath,
        param_filepath=param_filepath,
    )
    return yaml_filepath
