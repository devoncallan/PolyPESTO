from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from polypesto.core import petab as pet
from polypesto.core.params import ParameterSet
from polypesto.core.conditions import Conditions, SimConditions


@dataclass
class Dataset:
    """Container for experimental data and mapping to model observables.

    Attributes:
        `id` (str): Identifier for the dataset (e.g., filename or descriptive name).
        `data` (pd.DataFrame): DataFrame containing the experimental data.
        `tkey` (str): Column name in `data` representing time points (or independent variable).
        `obs_map` (Dict[str, str]): Mapping from DataFrame column names to model observable IDs.
            e.g., {"Conversion A": "xA", "Conversion B": "xB"}
    """

    id: str
    data: pd.DataFrame
    tkey: str
    obs_map: Dict[str, str]

    @staticmethod
    def load(path: str, tkey: str, obs_map: Dict[str, str], **kwargs) -> "Dataset":
        data = pd.read_csv(path, **kwargs)
        return Dataset(id=path, data=data, tkey=tkey, obs_map=obs_map)


@dataclass
class Experiment:
    """Container for data/metadata for a single experiment."""

    id: str
    conds: Conditions | SimConditions
    data: List[Dataset]

    @property
    def is_simulated(self) -> bool:
        return isinstance(self.conds, SimConditions)

    @staticmethod
    def load(id: str, conds: Dict[str, float], data: List[Dataset]) -> "Experiment":
        conditions = Conditions(exp_id=id, values=ParameterSet.lazy_from_dict(conds))
        return Experiment(id=id, conds=conditions, data=data)


def experiments_to_petab(experiments: List[Experiment]):

    data_dict = {}
    conds = []
    exp_ids = []
    for exp in experiments:

        cond = exp.conds
        exp_id = cond.exp_id
        conds.append(cond.values.to_dict())
        exp_ids.append(exp_id)

        for dataset in exp.data:

            t = dataset.data[dataset.tkey]
            for obs, col_name in dataset.obs_map.items():

                obs_id = f"obs_{obs}"
                y = dataset.data[col_name]
                key = (obs_id, exp_id)

                # Remove nans
                mask = ~np.isnan(y)
                t = t[mask]
                y = y[mask]

                if key in data_dict:
                    t_existing, y_existing = data_dict[key]
                    t = np.concatenate([t_existing, t])
                    y = np.concatenate([y_existing, y])

                data_dict[key] = (t, y)

    cond_df = pet.define_conditions(conds, exp_ids=exp_ids)
    meas_df = pet.define_measurements(data_dict)
    return cond_df, meas_df


def meas_df_to_datasets(meas_df: pd.DataFrame) -> List[Dataset]:

    obs_ids = meas_df[pet.C.OBSERVABLE_ID].unique()
    obs_map = {obs_id: obs_id for obs_id in obs_ids}

    wide = (
        meas_df.pivot(
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

    cond_df = petab_problem.condition_df
    meas_df = petab_problem.measurement_df

    assert cond_df is not None and meas_df is not None

    experiments = []
    exp_ids = meas_df[pet.C.SIMULATION_CONDITION_ID].unique()

    for exp_id in exp_ids:

        conds = cond_df[cond_df.index == exp_id].to_dict()
        exp_meas_df = meas_df[meas_df[pet.C.SIMULATION_CONDITION_ID] == exp_id]

        data = meas_df_to_datasets(exp_meas_df)
        exp = Experiment.load(id=exp_id, conds=conds, data=data)
        experiments.append(exp)

    return experiments


def modify_experiments(experiments: List[Experiment]) -> List[Experiment]:

    # Convert tkey in data to conversion
    # Add fA and fB to obs

    new_exps = []
    for exp in experiments:

        datasets = exp.data
        cond = exp.conds.values.to_dict()

        A0 = B0 = None
        if "A0" in cond and "B0" in cond:
            A0 = cond["A0"]
            B0 = cond["B0"]
        elif "fA0" in cond and "cM0" in cond:
            fA0 = cond["fA0"]
            cM0 = cond["cM0"]
            A0 = fA0 * cM0
            B0 = (1 - fA0) * cM0
        else:
            raise ValueError(
                f'Conditions must include either ("A0", "B0") or ("fA0", "cM0"). Actual: {list(cond.keys())}'
            )

        assert A0 is not None and B0 is not None
        assert A0 != 0 and B0 != 0

        fA0 = A0 / (A0 + B0)
        fB0 = B0 / (A0 + B0)

        new_datasets = []
        for ds in datasets:

            assert ds.tkey in ds.data.columns
            assert "xA" in ds.obs_map and "xB" in ds.obs_map

            ds.data[ds.tkey] = (
                fA0 * ds.data[ds.obs_map["xA"]] + fB0 * ds.data[ds.obs_map["xB"]]
            )

            mon_A = fA0 * (1 - ds.data[ds.obs_map["xA"]])
            mon_B = fB0 * (1 - ds.data[ds.obs_map["xB"]])
            ds.data["fA"] = mon_A / (mon_A + mon_B)
            ds.data["fB"] = mon_B / (mon_A + mon_B)

            ds.obs_map["fA"] = "fA"
            ds.obs_map["fB"] = "fB"

            new_datasets.append(ds)

        exp = Experiment(
            id=exp.id,
            conds=exp.conds,
            data=new_datasets,
        )
        new_exps.append(exp)

    return new_exps
