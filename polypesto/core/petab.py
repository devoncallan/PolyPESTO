from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeAlias

import numpy as np
import pandas as pd
import petab.v1.C as C  # type: ignore
from petab.v1 import Problem as PetabProblem  # type: ignore
from petab.v1 import (  # type: ignore
    write_condition_df,
    write_measurement_df,
    write_observable_df,
    write_parameter_df,
)
from petab.v1.lint import lint_problem  # type: ignore
from petab.v1.yaml import create_problem_yaml  # type: ignore


from polypesto.core.types import ModelDefinition
from polypesto.utils import ID, quiet


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
        self.obs_df = utils.obs.format(self.obs_df)
        self.cond_df = utils.cond.format(self.cond_df)
        self.param_df = utils.param.format(self.param_df)
        self.meas_df = utils.meas.format(self.meas_df)

        # Ensure noise formula (required in obs_df) does not reference noiseParameter
        # if no noise parameters are provided in meas_df
        # if C.NOISE_PARAMETERS not in self.meas_df.columns:
        #     values = [str(v) for v in self.obs_df[C.NOISE_FORMULA].values]
        #     if any("noiseParameter" in v for v in values):
        #         self.obs_df[C.NOISE_FORMULA] = [0.0] * len(values)

    def write(self, data_dir: str | Path, model: ModelDefinition) -> None:

        from polypesto.core.problem.core import ProblemPaths
        from polypesto.models.sbml import write_model

        paths = ProblemPaths(data_dir)

        write_model(model, paths.sbml_model)
        write_observable_df(self.obs_df, paths.observables)
        write_condition_df(self.cond_df, paths.conditions)
        write_parameter_df(self.param_df, paths.fit_parameters)
        write_measurement_df(self.meas_df, paths.measurements)

        create_problem_yaml(
            sbml_files=str(paths.sbml_model),
            condition_files=str(paths.conditions),
            measurement_files=str(paths.measurements),
            parameter_file=str(paths.fit_parameters),
            observable_files=str(paths.observables),
            yaml_file=str(paths.petab_yaml),
        )
        problem = PetabProblem.from_yaml(paths.petab_yaml)
        with quiet():
            lint_problem(problem)


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
    prior_type: Optional[str] = None
    prior_params: Optional[str] = None

    def set(
        self,
        scale: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
        nominal_value: Optional[float] = None,
        estimate: Optional[bool] = None,
        prior_type: Optional[str] = None,
        prior_params: Optional[str] = None,
    ):
        if scale is not None:
            self.scale = scale
        if bounds is not None:
            self.bounds = bounds
        if nominal_value is not None:
            self.nominal_value = nominal_value
        if estimate is not None:
            self.estimate = estimate
        if prior_type is not None:
            self.prior_type = prior_type
        if prior_params is not None:
            self.prior_params = prior_params


class utils:

    @staticmethod
    def check(df: pd.DataFrame, req_cols: List[str], raise_error: bool = True) -> bool:
        """Check if a DataFrame is a valid PEtab measurements DataFrame."""
        cols = set(df.columns.to_list()) | set([str(df.index.name)])
        is_valid = set(req_cols).issubset(cols)
        if raise_error and not is_valid:
            raise ValueError(
                f"DataFrame is missing required columns: {set(req_cols) - cols}"
            )
        return is_valid

    @staticmethod
    def format_df(
        df: pd.DataFrame, index_col: str, keep_column: bool = False
    ) -> pd.DataFrame:
        # Ensure the column is the index and not duplicated
        if df.index.name == index_col:  # If already indexed correctly, return as is
            return df
        elif index_col in df.columns:
            return df.set_index(index_col, inplace=False, drop=not keep_column)
        else:
            raise ValueError(f"Index column '{index_col}' not found in DataFrame.")

    class meas:

        @staticmethod
        def check(df: pd.DataFrame, **kwargs) -> bool:
            """Check if a DataFrame is a valid PEtab measurements DataFrame."""
            utils.check(df, C.MEASUREMENT_DF_REQUIRED_COLS, **kwargs)

        @staticmethod
        def format(df: pd.DataFrame) -> pd.DataFrame:
            """Format a DataFrame as a PEtab measurements DataFrame."""
            meas_df = utils.format_df(df, C.SIMULATION_CONDITION_ID, keep_column=True)
            utils.meas.check(meas_df)
            return meas_df

        @staticmethod
        def cond_ids(df: pd.DataFrame) -> List[str]:
            """Get unique condition IDs from a measurements DataFrame."""
            utils.meas.check(df)
            return [str(cid) for cid in df[C.SIMULATION_CONDITION_ID].unique()]

        @staticmethod
        def obs_ids(df: pd.DataFrame) -> List[str]:
            """Get unique observable IDs from a measurements DataFrame."""
            utils.meas.check(df)
            return [str(oid) for oid in df[C.OBSERVABLE_ID].unique()]

        @staticmethod
        def define(
            data_dict: Dict[ID.ObsCondKey, Tuple[np.ndarray, np.ndarray]],
            meas_noise_map: Dict[ID.ObsCondKey, float | np.ndarray] | None = None,
        ):
            """Define measurements DataFrame from a data dictionary.

            Args:
                data_dict (Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]): Mapping from (obs_id, cond_id) to (timepoints, measurements)
                noise_maps (Optional[Dict[Tuple[str, str], float]]): Optional mapping from (obs_id, cond_id) to noise values

            Returns:
                pd.DataFrame: Formatted measurements DataFrame
            """
            
            print("Defining measurements with utils.meas.define")  # Debug statement
            print(data_dict)
            print(meas_noise_map)

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
                    noise_val = meas_noise_map[key]
                    if isinstance(noise_val, np.ndarray):
                        if len(noise_val) != len(t):
                            raise ValueError(
                                f"Noise array for key {key} has length {len(noise_val)} but expected {len(t)}."
                            )
                        data[C.NOISE_PARAMETERS] = noise_val
                    else:
                        data[C.NOISE_PARAMETERS] = [float(noise_val)] * len(t)

                df = pd.DataFrame(data)
                meas_dfs.append(df)

            df = utils.meas.format(pd.concat(meas_dfs))
            utils.meas.check(df)
            return df

        @staticmethod
        def define_empty(
            data_dict: Dict[ID.ObsCondKey, np.ndarray], **kwargs
        ) -> pd.DataFrame:
            """Define empty measurements DataFrame from a data dictionary.

            Args:
                data_dict (Dict[ID.ObsCondKey, np.ndarray]): Mapping from (obs_id, cond_id) to timepoints

            Returns:
                pd.DataFrame: Formatted measurements DataFrame
            """

            empty_data_dict = {
                key: (t, np.zeros_like(t)) for key, t in data_dict.items()
            }
            return utils.meas.define(empty_data_dict, **kwargs)

        @staticmethod
        def add_noise(
            meas_df: pd.DataFrame,
            meas_noise: List[float] | List[Dict[ID.StrObsName, float]],
        ) -> pd.DataFrame:
            """Add Gaussian noise to the measurements DataFrame.

            Args:
                measurements_df: DataFrame with measurements
                noise_level: Standard deviation of the Gaussian noise
            """

            utils.meas.check(meas_df)

            noisy_meas_df = meas_df.copy()
            obs_ids = utils.meas.obs_ids(noisy_meas_df)
            cond_ids = utils.meas.cond_ids(noisy_meas_df)
            num_conds = len(cond_ids)

            if not isinstance(meas_noise, list) or len(meas_noise) != num_conds:
                raise ValueError(
                    f"meas_noise must be a list of length {num_conds}, got {type(meas_noise)} with length {len(meas_noise) if isinstance(meas_noise, list) else 'N/A'}"
                )

            for cond_id, noise in zip(cond_ids, meas_noise, strict=True):

                if isinstance(noise, dict):
                    meas_noise_dict: Dict[ID.StrObsID, float] = {
                        ID.obs_id(obs_name): noise_val
                        for obs_name, noise_val in noise.items()
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

                    values = noisy_meas_df.loc[mask, C.MEASUREMENT].values
                    noise_array = np.random.normal(0, noise_val * np.abs(values))

                    noisy_meas_df.loc[mask, C.MEASUREMENT] = values + noise_array

            return noisy_meas_df

    class obs:

        @staticmethod
        def check(df: pd.DataFrame, **kwargs) -> bool:
            """Check if a DataFrame is a valid PEtab parameters DataFrame."""
            utils.check(df, C.OBSERVABLE_DF_REQUIRED_COLS, **kwargs)

        @staticmethod
        def format(df: pd.DataFrame) -> pd.DataFrame:
            return utils.format_df(df, C.OBSERVABLE_ID, keep_column=False)

        @staticmethod
        def define(
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
            df = utils.obs.format(pd.DataFrame(data))
            utils.obs.check(df)
            return df

    class param:

        @staticmethod
        def check(df: pd.DataFrame, **kwargs) -> bool:
            """Check if a DataFrame is a valid PEtab parameters DataFrame."""
            utils.check(df, C.PARAMETER_DF_REQUIRED_COLS, **kwargs)

        @staticmethod
        def format(df: pd.DataFrame) -> pd.DataFrame:
            return utils.format_df(df, C.PARAMETER_ID, keep_column=False)

        @staticmethod
        def define(params: Dict[str, FitParameter]) -> pd.DataFrame:
            data = []
            for param in params.values():
                row = {
                    C.PARAMETER_ID: param.id,
                    C.PARAMETER_SCALE: param.scale,
                    C.LOWER_BOUND: param.bounds[0],
                    C.UPPER_BOUND: param.bounds[1],
                    C.NOMINAL_VALUE: param.nominal_value,
                    C.ESTIMATE: param.estimate,
                }
                # Add prior columns if specified
                if param.prior_type is not None:
                    row[C.OBJECTIVE_PRIOR_TYPE] = param.prior_type
                if param.prior_params is not None:
                    row[C.OBJECTIVE_PRIOR_PARAMETERS] = param.prior_params
                data.append(row)

            df = pd.DataFrame(data)
            df = utils.param.format(df)
            utils.param.check(df)
            return df

    class cond:

        @staticmethod
        def format(df: pd.DataFrame) -> pd.DataFrame:
            return utils.format_df(df, C.CONDITION_ID, keep_column=False)

        @staticmethod
        def define(
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
                print([c.keys() for c in conds])
                raise ValueError("All condition dictionaries must have the same keys")

            df = pd.DataFrame(conds)
            df[C.CONDITION_ID] = ids
            df[C.CONDITION_NAME] = names

            df = utils.cond.format(df)
            return df

__all__ = [
    "C",
    "PetabProblem",
    "PetabData",
    "FitParameter",
    "define_parameters",
    "define_observables",
    "define_conditions",
    "define_measurements",
    "define_empty_measurements",
    "add_noise_to_measurements",
]
