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


class PetabIO:

    @staticmethod
    def format_df(
        df: pd.DataFrame, index_col: str, keep_column: bool = False
    ) -> pd.DataFrame:
        # Ensure the column is the index and not duplicated
        if df.index.name == index_col:
            # If already indexed correctly, return as is
            return df
        elif index_col in df.columns:
            return df.set_index(index_col, inplace=False, drop=not keep_column)
        else:
            raise ValueError(f"Index column '{index_col}' not found in DataFrame.")

    @staticmethod
    def format_obs_df(df: pd.DataFrame) -> pd.DataFrame:
        return PetabIO.format_df(df, C.OBSERVABLE_ID, keep_column=False)

    @staticmethod
    def format_cond_df(df: pd.DataFrame) -> pd.DataFrame:
        return PetabIO.format_df(df, C.CONDITION_ID, keep_column=False)

    @staticmethod
    def format_meas_df(df: pd.DataFrame) -> pd.DataFrame:
        return PetabIO.format_df(df, C.SIMULATION_CONDITION_ID, keep_column=True)

    @staticmethod
    def format_param_df(df: pd.DataFrame) -> pd.DataFrame:
        return PetabIO.format_df(df, C.PARAMETER_ID, keep_column=False)

    @staticmethod
    def read_petab_df(filepath: str, format_func) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep="\t")  # .reset_index(drop=True)
        return format_func(df)

    @staticmethod
    def read_obs_df(filepath: str) -> pd.DataFrame:
        return PetabIO.read_petab_df(filepath, PetabIO.format_obs_df)

    @staticmethod
    def read_cond_df(filepath: str) -> pd.DataFrame:
        return PetabIO.read_petab_df(filepath, PetabIO.format_cond_df)

    @staticmethod
    def read_meas_df(filepath: str) -> pd.DataFrame:
        return PetabIO.read_petab_df(filepath, PetabIO.format_meas_df)

    @staticmethod
    def read_param_df(filepath: str) -> pd.DataFrame:
        return PetabIO.read_petab_df(filepath, PetabIO.format_param_df)

    @staticmethod
    def write_df(df: pd.DataFrame, write_func, dir: str = None, filename: str = None):
        filepath = os.path.join(dir, filename) if dir else filename
        write_func(df, filepath)
        return filepath

    @staticmethod
    def write_obs_df(
        df: pd.DataFrame, dir: str = None, filename: str = "observables.tsv"
    ):
        return PetabIO.write_df(df, petab.v1.write_observable_df, dir, filename)

    @staticmethod
    def write_cond_df(
        df: pd.DataFrame, dir: str = None, filename: str = "conditions.tsv"
    ):
        return PetabIO.write_df(df, petab.v1.write_condition_df, dir, filename)

    @staticmethod
    def write_meas_df(
        df: pd.DataFrame, dir: str = None, filename: str = "measurements.tsv"
    ):
        return PetabIO.write_df(df, petab.v1.write_measurement_df, dir, filename)

    @staticmethod
    def write_param_df(
        df: pd.DataFrame, dir: str = None, filename: str = "parameters.tsv"
    ):
        return PetabIO.write_df(df, petab.v1.write_parameter_df, dir, filename)


def define_parameters(params_dict: Dict[str, FitParameter]) -> pd.DataFrame:
    df = pd.DataFrame(
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
    )  # .set_index(C.PARAMETER_ID)
    return PetabIO.format_param_df(df)


def define_observables(
    observables: Dict[str, str], noise_value: float = 0.0
) -> pd.DataFrame:

    observable_ids = list(observables.keys())
    observable_formulas = list(observables.values())

    df = pd.DataFrame(
        data={
            C.OBSERVABLE_ID: [f"obs_{id}" for id in observable_ids],
            C.OBSERVABLE_FORMULA: observable_formulas,
            C.NOISE_FORMULA: [noise_value] * len(observable_ids),
        }
    )  # .set_index(C.OBSERVABLE_ID)
    return PetabIO.format_obs_df(df)


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

    return PetabIO.format_cond_df(conditions)


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

    obs_filepath = PetabIO.write_obs_df(obs_df, dir=obs_dir)
    cond_filepath = PetabIO.write_cond_df(cond_df, dir=cond_dir)
    meas_filepath = PetabIO.write_meas_df(meas_df, dir=meas_dir)
    param_filepath = PetabIO.write_param_df(param_df, dir=param_dir)

    yaml_filepath = write_yaml_file(
        yaml_dir,
        sbml_filepath,
        cond_filepath=cond_filepath,
        meas_filepath=meas_filepath,
        obs_filepath=obs_filepath,
        param_filepath=param_filepath,
    )
    return yaml_filepath
