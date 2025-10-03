from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeAlias
from uuid import uuid4

import numpy as np
import pandas as pd

from polypesto.utils import ID

from . import petab as pet
from .params import ParameterSet


@dataclass
class Dataset:
    """Container for experimental data and mapping to model observables.

    Attributes:
        `id` (str): Identifier for the dataset (e.g., filename or descriptive name).
        `data` (pd.DataFrame): DataFrame containing the experimental data.
        `tkey` (str): Column name in `data` representing time points (or independent variable).
        `obs_map` (Dict[ID.StrObsName, str]): Mapping from DataFrame column names to model observable names.
            e.g., {"xA": "Conversion A", "xB": "Conversion B"}
        `noise_map` (Optional[Dict[ID.StrObsName, float]]): Optional mapping from observable names to noise
            e.g., {"xA": 0.1, "xB": 0.2}
    """

    id: str
    data: pd.DataFrame
    tkey: str
    obs_map: Dict[ID.StrObsName, str]
    noise_map: Optional[Dict[ID.StrObsName, float]] = None

    def __post_init__(self):
        """Validate that tkey and obs_map columns exist in the data."""

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(self.data)}")

        if len(self.data) == 0:
            raise ValueError("Provided DataFrame is empty.")

        if self.tkey not in self.data.columns:
            raise KeyError(
                f"Time key '{self.tkey}' not found in data columns ({self.data.columns.tolist()})."
            )

        missing_cols = [
            col for col in self.obs_map.values() if col not in self.data.columns
        ]
        if missing_cols:
            raise KeyError(
                f"Observable columns {missing_cols} not found in data columns ({self.data.columns.tolist()})."
            )

    @staticmethod
    def load(
        path_or_data: str | pd.DataFrame,
        tkey: str,
        obs_map: Dict[ID.StrObsName, str],
        noise_map: Optional[Dict[ID.StrObsName, float]] = None,
        **kwargs,
    ) -> Dataset:
        if isinstance(path_or_data, pd.DataFrame):
            id = str(uuid4())
            data = path_or_data
        else:
            id = str(path_or_data)
            data = pd.read_csv(path_or_data, **kwargs)

        return Dataset(
            id=id, data=data, tkey=tkey, obs_map=obs_map, noise_map=noise_map
        )


@dataclass
class Experiment:
    """Container for data/metadata for a single experiment."""

    id: str
    conds: ParameterSet
    data: List[Dataset]

    @staticmethod
    def load(
        id: str, conds: Dict[ID.StrCondName, float], data: List[Dataset]
    ) -> Experiment:
        conditions = ParameterSet.from_dict(conds, id=id)
        return Experiment(id=id, conds=conditions, data=data)


def experiments_to_petab(
    experiments: List[Experiment],
    obs_noise_map: Dict[ID.StrObsName, float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a list of Experiment objects to PEtab format.

    Args:
        experiments (List[Experiment]): List of Experiment objects.
        obs_noise_map (Optional[Dict[ID.StrObsName, float]]): Optional mapping from observable names to noise
            parameters to override dataset-specific noise maps. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: PEtab conditions and measurements dataframes.
    """

    data_dict: Dict[ID.ObsCondKey, Tuple[np.ndarray, np.ndarray]] = {}
    noise_map: Dict[ID.ObsCondKey, float] = {}

    conds = []

    cond_names = [str(exp.conds.id) for exp in experiments]
    cond_ids = [ID.cond_id(name) for name in cond_names]

    if len(cond_ids) != len(set(cond_ids)):
        raise ValueError(
            f"Condition IDs must be unique. Found duplicates in {cond_ids}"
        )

    for i, exp in enumerate(experiments):
        cond = exp.conds
        conds.append(cond.to_dict())

        for dataset in exp.data:

            for obs_name, col_name in dataset.obs_map.items():

                key = (ID.obs_id(obs_name), cond_ids[i])
                t = np.array(dataset.data[dataset.tkey])
                y = np.array(dataset.data[col_name])

                # Remove nans
                mask = ~np.isnan(y)
                t = t[mask]
                y = y[mask]

                if key in data_dict:
                    t_existing, y_existing = data_dict[key]
                    t = np.concatenate([t_existing, t])
                    y = np.concatenate([y_existing, y])

                data_dict[key] = (t, y)

                # Override noise map from observables if in experiments
                if dataset.noise_map and obs_name in dataset.noise_map:
                    noise_map[key] = dataset.noise_map[obs_name]
                elif obs_noise_map and obs_name in obs_noise_map:
                    noise_map[key] = obs_noise_map[obs_name]
                else:
                    noise_map[key] = 0.0

    if all(v == 0.0 for v in noise_map.values()):
        noise_map = None

    cond_df = pet.utils.cond.define(conds, names=cond_names)
    meas_df = pet.utils.meas.define(data_dict, noise_map)
    return cond_df, meas_df


def petab_to_experiments(petab_problem: pet.PetabProblem) -> List[Experiment]:
    """Convert a PEtab problem to a list of Experiment objects.

    Args:
        petab_problem (pet.PetabProblem): PEtab problem instance.

    Returns:
        List[Experiment]: List of Experiment objects.
    """

    cond_df = petab_problem.condition_df
    meas_df = petab_problem.measurement_df
    if cond_df is None or meas_df is None:
        raise ValueError(
            "PEtab problem must have condition and measurement dataframes."
        )

    cond_ids = pet.utils.meas.cond_ids(meas_df)
    cond_dict = cond_df.drop(columns=pet.C.CONDITION_NAME).to_dict(orient="index")

    #
    experiments = []
    for cond_id in cond_ids:

        conds = cond_dict[cond_id]

        exp_meas_df = meas_df[meas_df[pet.C.SIMULATION_CONDITION_ID] == cond_id]

        obs_ids = pet.utils.meas.obs_ids(exp_meas_df)

        # Assume formula is just the observable ID for now
        obs_map = {obs_id: obs_id for obs_id in obs_ids}

        noise_map = None
        if pet.C.NOISE_PARAMETERS in exp_meas_df.columns:
            noise_map = {
                obs_id: float(
                    exp_meas_df[exp_meas_df[pet.C.OBSERVABLE_ID] == obs_id][
                        pet.C.NOISE_PARAMETERS
                    ].unique()[0]
                )  # take first unique value
                for obs_id in obs_ids
            }

        exp_wide_df = (
            exp_meas_df.pivot_table(
                index=pet.C.TIME,
                columns=pet.C.OBSERVABLE_ID,
                values=pet.C.MEASUREMENT,
            )
            .rename(columns=obs_map)
            .reset_index()
            .sort_values(pet.C.TIME)
        )
        exp_wide_df.columns.name = None

        data = Dataset(
            id=f"Dataset_for_{cond_id}",
            data=exp_wide_df,
            tkey=pet.C.TIME,
            obs_map=obs_map,
            noise_map=noise_map,
        )

        exp = Experiment.load(cond_id, conds, [data])
        experiments.append(exp)

    return experiments
