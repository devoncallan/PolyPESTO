import os
from typing import Dict, Tuple, Sequence, Callable, Optional
from functools import wraps
from dataclasses import dataclass

import numpy as np
import pandas as pd
import petab
import petab.v1.C as C


@dataclass
class PetabData:
    """
    Simple data container for grouping PEtab dataframes together.
    """

    obs_df: pd.DataFrame
    cond_df: pd.DataFrame
    param_df: pd.DataFrame
    meas_df: pd.DataFrame
    name: str = None


@dataclass
class FitParameter:
    """
    Simple data container for defining a PEtab fit parameter.
    """

    id: str
    scale: str
    bounds: Tuple[float, float]
    nominal_value: float
    estimate: bool


class PetabIO:
    """
    Namespace for reading and writing PEtab files.
    """

    ##########################
    ### Format PETab files ###
    ##########################

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

    ########################
    ### Read PETab files ###
    ########################

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

    #########################
    ### Write PETab files ###
    #########################

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

    @staticmethod
    def write_yaml(
        yaml_filepath: str,
        sbml_filepath: str,
        cond_filepath: str,
        meas_filepath: str,
        obs_filepath: str,
        param_filepath: str,
    ) -> str:

        petab.v1.yaml.create_problem_yaml(
            sbml_files=sbml_filepath,
            condition_files=cond_filepath,
            measurement_files=meas_filepath,
            parameter_file=param_filepath,
            observable_files=obs_filepath,
            yaml_file=yaml_filepath,
            relative_paths=False,
        )
        problem = petab.v1.Problem.from_yaml(yaml_filepath, base_path="")
        petab.v1.lint.lint_problem(problem)

        return yaml_filepath


############################
### Define petab problem ###
############################


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
    )
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
    )
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


def define_empty_measurements(
    observables_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    timepoints: Sequence[float] | Sequence[Sequence[float]] = None,
) -> pd.DataFrame:
    """Create empty measurements DataFrame with specified timepoints.

    Args:
        observables_df: DataFrame with observable definitions
        conditions_df: DataFrame with condition definitions
        timepoints: Either:
            - Sequence[float]: Same timepoints used for all conditions
            - Sequence[Sequence[float]]: Different timepoints per condition
            - None: Defaults to [0] for all conditions
    """
    if timepoints is None:
        timepoints = [0]

    # If single sequence provided, use for all conditions
    if not isinstance(timepoints[0], (list, tuple)):
        timepoints = [timepoints] * len(conditions_df)

    meas_dfs = []
    for (cond_id, _), cond_times in zip(conditions_df.iterrows(), timepoints):
        cond_meas_dfs = []
        for obs_id, _ in observables_df.iterrows():
            df = pd.DataFrame(
                {
                    C.OBSERVABLE_ID: [obs_id] * len(cond_times),
                    C.SIMULATION_CONDITION_ID: [cond_id] * len(cond_times),
                    C.TIME: cond_times,
                    C.MEASUREMENT: [0] * len(cond_times),
                }
            )
            cond_meas_dfs.append(df)
        meas_df = pd.concat(cond_meas_dfs)
        meas_dfs.append(meas_df)

    meas_df = pd.concat(meas_dfs)
    return PetabIO.format_meas_df(meas_df)
    # return PetabIO.format_meas_df(meas_df)


def add_noise_to_measurements(
    measurements_df: pd.DataFrame,
    noise_level: float,
) -> pd.DataFrame:
    """Add Gaussian noise to the measurements DataFrame.

    Args:
        measurements_df: DataFrame with measurements
        noise_level: Standard deviation of the Gaussian noise
    """
    noisy_measurements = measurements_df.copy()
    values = noisy_measurements[C.MEASUREMENT].values

    noise = np.random.normal(0, noise_level * np.abs(values))
    noisy_measurements[C.MEASUREMENT] = values + noise

    return noisy_measurements
