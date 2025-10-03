from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeAlias

import numpy as np
import pandas as pd
import petab.v1.C as C  # type: ignore
from petab.v1 import Problem as PetabProblem  # type: ignore
from petab.v1 import (
    write_condition_df,
    write_measurement_df,
    write_observable_df,
    write_parameter_df,
)
from petab.v1.lint import lint_problem  # type: ignore
from petab.v1.yaml import create_problem_yaml  # type: ignore

from polypesto.utils import ID


@dataclass
class PetabData:
    """
    Simple data container for grouping PEtab dataframes together.
    """

    obs_df: pd.DataFrame
    cond_df: pd.DataFrame
    param_df: pd.DataFrame
    meas_df: pd.DataFrame
    name: Optional[str] = None

    def __post_init__(self):
        """Validate that the dataframes have the correct format."""
        self.obs_df = PetabIO.format_obs_df(self.obs_df)
        self.cond_df = PetabIO.format_cond_df(self.cond_df)
        self.param_df = PetabIO.format_param_df(self.param_df)
        self.meas_df = PetabIO.format_meas_df(self.meas_df)

        # Ensure noise formula (required in obs_df) does not reference noiseParameter 
        # if no noise parameters are provided in meas_df
        if C.NOISE_PARAMETERS not in self.meas_df.columns:
            values = [str(v) for v in self.obs_df[C.NOISE_FORMULA].values]
            if any("noiseParameter" in v for v in values):
                self.obs_df[C.NOISE_FORMULA] = [0.0] * len(values)


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
    _WriteFunc: TypeAlias = Callable[[pd.DataFrame, str | Path], None]

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
    def write_yaml(
        yaml_filepath: str | Path,
        sbml_filepath: str | Path,
        cond_filepath: str | Path,
        meas_filepath: str | Path,
        obs_filepath: str | Path,
        param_filepath: str | Path,
    ) -> Path:

        create_problem_yaml(
            sbml_files=str(sbml_filepath),
            condition_files=str(cond_filepath),
            measurement_files=str(meas_filepath),
            parameter_file=str(param_filepath),
            observable_files=str(obs_filepath),
            yaml_file=str(yaml_filepath),
        )
        problem = PetabProblem.from_yaml(yaml_filepath)
        lint_problem(problem)

        return Path(yaml_filepath)


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
    obs_formula_map: Dict[ID.StrObsName, ID.StrObsFormula],
    obs_noise_map: Dict[ID.StrObsName, float] | None = None,
) -> pd.DataFrame:

    obs_names = list(obs_formula_map.keys())
    obs_formulas = list(obs_formula_map.values())
    obs_ids = [ID.obs_id(name) for name in obs_names]

    if obs_noise_map:
        if set(obs_names) != set(obs_noise_map.keys()):
            raise ValueError(
                "Observable names in obs_formula_map and obs_noise_map must match."
            )
        noise_formulas = [obs_noise_map[name] for name in obs_names]
    else:
        noise_formulas = [f"noiseParameter1_{obs_id}" for obs_id in obs_ids]

    data = {
        C.OBSERVABLE_ID: obs_ids,
        C.OBSERVABLE_NAME: obs_names,
        C.OBSERVABLE_FORMULA: obs_formulas,
        C.NOISE_FORMULA: noise_formulas,
    }

    df = pd.DataFrame(data)
    return PetabIO.format_obs_df(df)


def define_conditions(
    conds: List[Dict[ID.StrCondName, float]],
    names: Optional[List[ID.StrCondID]] = None,
    ids: Optional[List[ID.StrCondID]] = None,
) -> pd.DataFrame:

    if names is None and ids is not None:
        names = ids
    elif ids is None and names is not None:
        ids = [ID.cond_id(name) for name in names]
    else:
        ids = ID.make_cond_ids(len(conds))
        names = ids

    if len(ids) != len(conds) or len(names) != len(conds):
        raise ValueError(
            f"Number of provided cond_ids ({len(ids)}) must match number of conditions ({len(conds)})."
        )

    if len({frozenset(c.keys()) for c in conds}) != 1:
        raise ValueError("All condition dictionaries must have the same keys")

    df = pd.DataFrame(conds)
    df[C.CONDITION_ID] = ids
    df[C.CONDITION_NAME] = names

    return PetabIO.format_cond_df(df)


def define_measurements(
    data_dict: Dict[ID.ObsCondKey, Tuple[np.ndarray, np.ndarray]],
    meas_noise_map: Dict[ID.ObsCondKey, float] | None = None,
):
    """Define measurements DataFrame from a data dictionary.

    Args:
        data_dict (Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]): Mapping from (obs_id, cond_id) to (timepoints, measurements)
        noise_maps (Optional[Dict[Tuple[str, str], float]]): Optional mapping from (obs_id, cond_id) to noise values

    Returns:
        pd.DataFrame: Formatted measurements DataFrame
    """

    if meas_noise_map is not None:
        if set(data_dict.keys()) != set(meas_noise_map.keys()):
            raise ValueError(
                "Keys of data_dict and meas_noise_map must match if meas_noise_map is provided."
            )

    meas_dfs = []
    for key, (t, y) in data_dict.items():

        obs_id, cond_id = key

        data = {
            C.OBSERVABLE_ID: obs_id,
            C.SIMULATION_CONDITION_ID: cond_id,
            C.TIME: t,
            C.MEASUREMENT: y,
        }

        if meas_noise_map and key in meas_noise_map:
            data[C.NOISE_PARAMETERS] = [meas_noise_map[key]] * len(t)

        df = pd.DataFrame(data)
        meas_dfs.append(df)

    meas_df = pd.concat(meas_dfs)
    return PetabIO.format_meas_df(meas_df)


def define_empty_measurements(
    data_dict: Dict[ID.ObsCondKey, np.ndarray], **kwargs
) -> pd.DataFrame:
    """Define empty measurements DataFrame from a data dictionary.

    Args:
        data_dict (Dict[ID.ObsCondKey, np.ndarray]): Mapping from (obs_id, cond_id) to timepoints

    Returns:
        pd.DataFrame: Formatted measurements DataFrame
    """

    empty_data_dict = {key: (t, np.zeros_like(t)) for key, t in data_dict.items()}
    return define_measurements(empty_data_dict, **kwargs)


def add_noise_to_measurements(
    meas_df: pd.DataFrame,
    meas_noise: List[float] | List[Dict[ID.StrObsName, float]],
) -> pd.DataFrame:
    """Add Gaussian noise to the measurements DataFrame.

    Args:
        measurements_df: DataFrame with measurements
        noise_level: Standard deviation of the Gaussian noise
    """

    noisy_meas_df = meas_df.copy()
    obs_ids = list(set(noisy_meas_df[C.OBSERVABLE_ID].values))
    obs_ids = [str(obs_id) for obs_id in obs_ids]

    cond_ids = list(set(noisy_meas_df[C.SIMULATION_CONDITION_ID].values))
    cond_ids = [str(cond_id) for cond_id in cond_ids]
    num_conds = len(cond_ids)

    if not isinstance(meas_noise, list) or len(meas_noise) != num_conds:
        raise ValueError(
            f"meas_noise must be a list of length {num_conds}, got {type(meas_noise)} with length {len(meas_noise) if isinstance(meas_noise, list) else 'N/A'}"
        )

    for cond_id, noise in zip(cond_ids, meas_noise, strict=True):

        if isinstance(noise, dict):
            meas_noise_dict: Dict[ID.StrObsID, float] = {
                ID.obs_id(obs_name): noise_val for obs_name, noise_val in noise.items()
            }
        elif isinstance(noise, (int, float)):
            meas_noise_dict: Dict[ID.StrObsID, float] = {
                obs_id: float(noise) for obs_id in obs_ids
            }
        else:
            raise TypeError("meas_noise must be a float or a dict")

        for obs_id, noise_val in meas_noise_dict.items():

            mask = (noisy_meas_df[C.OBSERVABLE_ID] == obs_id) & (
                noisy_meas_df[C.SIMULATION_CONDITION_ID] == cond_id
            )

            if not mask.any():
                print(
                    f"Warning: No measurements found for observable '{obs_id}' and condition '{cond_id}'"
                )
                continue

            # values = noisy_meas_df[mask][C.MEASUREMENT].values
            values = noisy_meas_df.loc[mask, C.MEASUREMENT].values
            noise_array = np.random.normal(0, noise_val * np.abs(values))

            noisy_meas_df.loc[mask, C.MEASUREMENT] = values + noise_array

    return noisy_meas_df


__all__ = [
    "C",
    "write_observable_df",
    "write_condition_df",
    "write_measurement_df",
    "write_parameter_df",
    "PetabProblem",
    "PetabData",
    "FitParameter",
    "PetabIO",
    "define_parameters",
    "define_observables",
    "define_conditions",
    "define_measurements",
    "define_empty_measurements",
    "add_noise_to_measurements",
]
