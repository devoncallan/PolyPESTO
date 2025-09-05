import os
from typing import Dict, List, Optional, Tuple, Sequence, Callable, TypeAlias
from dataclasses import dataclass


import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import petab
import petab.v1.C as C
from petab.v1 import Problem as PetabProblem


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

    def set(
        self,
        scale: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
        nominal_value: Optional[float] = None,
        estimate: Optional[bool] = None,
    ):
        if scale is not None:
            self.scale = scale
        if bounds is not None:
            self.bounds = bounds
        if nominal_value is not None:
            self.nominal_value = nominal_value
        if estimate is not None:
            self.estimate = estimate


class PetabIO:
    """
    Namespace for reading and writing PEtab files.
    """

    _FormatFunc: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]

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
    def read_petab_df(filepath: str, format_func: _FormatFunc) -> pd.DataFrame:
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


def define_conditions(
    conds: List[Dict[str, float]], exp_ids: List[str] = None
) -> pd.DataFrame:

    if exp_ids is None:
        exp_ids = [f"exp_{i}" for i in range(len(conds))]
    elif len(exp_ids) != len(conds):
        raise ValueError(
            f"Number of provided exp_ids ({len(exp_ids)}) must match number of conditions ({len(conds)})."
        )

    if len({frozenset(c.keys()) for c in conds}) != 1:
        raise ValueError("All condition dictionaries must have the same keys")

    df = pd.DataFrame(conds)
    df[C.CONDITION_ID] = exp_ids
    df[C.CONDITION_NAME] = exp_ids

    return PetabIO.format_cond_df(df)


def define_measurements(
    data_dict: Dict[Tuple[str, str], Tuple[ArrayLike, ArrayLike]],
):
    """Define measurements DataFrame from a data dictionary.

    Args:
        data_dict (Dict[Tuple[str, str], Tuple[ArrayLike, ArrayLike]]): Mapping from (observable_id, exp_id) to (timepoints, measurements)

    Returns:
        pd.DataFrame: Formatted measurements DataFrame
    """

    meas_dfs = []
    for (obs_id, cond_id), (t, y) in data_dict.items():
        df = pd.DataFrame(
            {
                C.OBSERVABLE_ID: [obs_id] * len(t),
                C.SIMULATION_CONDITION_ID: [cond_id] * len(t),
                C.TIME: t,
                C.MEASUREMENT: y,
            }
        )
        meas_dfs.append(df)
    meas_df = pd.concat(meas_dfs)
    return PetabIO.format_meas_df(meas_df)


def define_empty_measurements(
    data_dict: Dict[Tuple[str, str], ArrayLike],
) -> pd.DataFrame:

    data_dict = {
        (obs_id, cond_id): (t, np.zeros_like(t))
        for (obs_id, cond_id), t in data_dict.items()
    }
    return define_measurements(data_dict)


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
