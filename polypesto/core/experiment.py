from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
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
        `obs_map` (Dict[str, str]): Mapping from DataFrame column names to model observable IDs.
            e.g., {"xA": "Conversion A", "xB": "Conversion B"}
    """

    id: str
    data: pd.DataFrame
    tkey: str
    obs_map: Dict[str, str]

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
        path_or_data: str | pd.DataFrame, tkey: str, obs_map: Dict[str, str], **kwargs
    ) -> "Dataset":
        if isinstance(path_or_data, pd.DataFrame):
            id = str(uuid4())
            data = path_or_data
        else:
            id = str(path_or_data)
            data = pd.read_csv(path_or_data, **kwargs)

        return Dataset(id=id, data=data, tkey=tkey, obs_map=obs_map)


@dataclass
class Experiment:
    """Container for data/metadata for a single experiment."""

    id: str
    conds: ParameterSet
    data: List[Dataset]

    @staticmethod
    def load(id: str, conds: Dict[str, float], data: List[Dataset]) -> Experiment:
        conditions = ParameterSet.from_dict(conds, id=id)
        return Experiment(id=id, conds=conditions, data=data)


def experiments_to_petab(
    experiments: List[Experiment],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a list of Experiment objects to PEtab format.

    Args:
        experiments (List[Experiment]): List of Experiment objects.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: PEtab conditions and measurements dataframes.
    """

    data_dict: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    conds = []

    cond_ids = [ID.cond_id(exp.conds.id) for exp in experiments]
    assert len(cond_ids) == len(
        set(cond_ids)
    ), f"Condition IDs must be unique. Found duplicates in {cond_ids}"

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

    cond_df = pet.define_conditions(conds, cond_ids)
    meas_df = pet.define_measurements(data_dict)
    return cond_df, meas_df


def meas_df_to_datasets(meas_df: pd.DataFrame) -> List[Dataset]:
    """Convert a PEtab measurement dataframe to a list of Dataset objects.

    Args:
        meas_df (pd.DataFrame): PEtab measurement dataframe.

    Returns:
        List[Dataset]: List of Dataset objects.
    """

    obs_ids = meas_df[pet.C.OBSERVABLE_ID].unique()
    obs_map = {obs_id: obs_id for obs_id in obs_ids}

    wide = (
        meas_df.pivot_table(
            index=pet.C.TIME,
            columns=pet.C.OBSERVABLE_ID,
            values=pet.C.MEASUREMENT,
        )
        .rename(columns=obs_map)
        .reset_index()
        .sort_values(pet.C.TIME)
    )
    wide.columns.name = None

    cond_id = str()

    ds = Dataset(
        id=f"Dataset_for_{cond_id}",
        data=wide,
        tkey=pet.C.TIME,
        obs_map=obs_map,
    )
    return [ds]


def petab_to_experiments(petab_problem: pet.PetabProblem) -> List[Experiment]:
    """Convert a PEtab problem to a list of Experiment objects.

    Args:
        petab_problem (pet.PetabProblem): PEtab problem instance.

    Returns:
        List[Experiment]: List of Experiment objects.
    """

    cond_df = petab_problem.condition_df
    meas_df = petab_problem.measurement_df

    assert cond_df is not None and meas_df is not None

    experiments = []
    cond_ids = meas_df[pet.C.SIMULATION_CONDITION_ID].unique()

    cond_dict = cond_df.drop(columns=pet.C.CONDITION_NAME).to_dict(orient="index")

    for cond_id in cond_ids:

        conds = cond_dict[cond_id]

        exp_meas_df = meas_df[meas_df[pet.C.SIMULATION_CONDITION_ID] == cond_id]

        data = meas_df_to_datasets(exp_meas_df)
        exp = Experiment.load(cond_id, conds, data)
        experiments.append(exp)

    return experiments
