from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from polypesto.core import petab as pet
from polypesto.core.params import ParameterSet


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
class Conditions:
    """Initial conditions for a given experiment."""

    exp_id: str
    values: ParameterSet


@dataclass
class SimConditions(Conditions):
    """Simulation conditions for a given experiment."""

    true_params: ParameterSet
    t_eval: ArrayLike
    noise_level: float = 0.0


def create_conditions(
    conds: Dict[str, ArrayLike], exp_ids: Optional[List[str]] = None
) -> List[Conditions]:

    cond_labels = list(conds.keys())
    len_conds = {k: len(v) for k, v in conds.items()}
    n_conds = len_conds[cond_labels[0]]

    if not np.all(np.array(list(len_conds.values())) == n_conds):
        raise ValueError(
            f"All condition lists must have the same length. Lengths: {len_conds}"
        )

    if exp_ids is None:
        exp_ids = [f"c_{i}" for i in range(n_conds)]
    elif len(exp_ids) != n_conds:
        raise ValueError(
            f"Length of exp_ids ({len(exp_ids)}) must match number of conditions ({n_conds})."
        )

    conditions = []
    for i in range(n_conds):
        cond = Conditions(
            exp_id=exp_ids[i],
            values=ParameterSet.lazy_from_dict(
                {cond_id: conds[cond_id][i] for cond_id in cond_labels}, id=exp_ids[i]
            ),
        )
        conditions.append(cond)

    return conditions


def create_sim_conditions(
    true_params: ParameterSet | Dict[str, float],
    conds: Dict[str, ArrayLike],
    t_evals: ArrayLike | List[ArrayLike],
    noise_levels: float | List[float] = 0.0,
    exp_ids: Optional[List[str]] = None,
) -> List[SimConditions]:

    if isinstance(true_params, dict):
        true_params = ParameterSet.lazy_from_dict(true_params)
    elif not isinstance(true_params, ParameterSet):
        raise ValueError("true_params must be a ParameterSet or a dict.")

    conditions = create_conditions(conds, exp_ids)
    n_conds = len(conditions)

    if isinstance(t_evals, np.ndarray):
        t_evals = [t_evals] * n_conds
    elif len(t_evals) != n_conds:
        raise ValueError(
            f"Length of t_evals ({len(t_evals)}) must match number of conditions ({n_conds})."
        )
    assert isinstance(t_evals, list) and len(t_evals) == n_conds

    if isinstance(noise_levels, float):
        noise_levels = [noise_levels] * n_conds
    elif len(noise_levels) != n_conds:
        raise ValueError(
            f"Length of noise_levels ({len(noise_levels)}) must match number of conditions ({n_conds})."
        )
    assert isinstance(noise_levels, list) and len(noise_levels) == n_conds

    sim_conditions = []
    for i in range(n_conds):

        sim_cond = SimConditions(
            exp_id=conditions[i].exp_id,
            values=conditions[i].values,
            true_params=true_params,
            t_eval=t_evals[i],
            noise_level=noise_levels[i],
        )
        sim_conditions.append(sim_cond)

    return sim_conditions


def define_cond_df(conds: List[Conditions]) -> pd.DataFrame:
    """Define a PEtab conditions dataframe from a list of Conditions."""

    return pet.define_conditions(
        [cond.values.to_dict() for cond in conds],
        exp_ids=[cond.exp_id for cond in conds],
    )


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


def create_petab_from_experiments(experiments: List[Experiment]):

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


from polypesto.core.experiment import Dataset, Experiment


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


def create_experiments_from_petab(petab_problem: pet.PetabProblem) -> List[Experiment]:

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


"""

from polypesto.models import IrreversibleCPE

model = IrreversibleCPE(
    data_dir="path/to/data",
    obs=["xA", "xB", "fA", "fB"]
)
model.fit_params["rA"].estimate = False

exp1 = Experiment(
    name="30:70 ELp:MMA",
    conds={"fA0": 0.30, "cM0": 1.0},
    data=[
        Dataset.load(
            path="path/to/data.csv",
            tkey="time",
            obs_map={"xA": "Conversion ELp", "xB": "Conversion MMA"}
        ),
    ]
)

exp2 = Experiment(
    name="50:50 ELp:MMA",
    conds={"fA0": 0.50, "cM0": 1.0},
    data=[
        Dataset.load(
            path="path/to/data.csv",
            tkey="time",
            obs_map={"xA": "Conversion ELp", "xB": "Conversion MMA"}
        ),
    ]
)

problem = Problem[Model].from_experiments([exp1, exp2])

run_parameter_estimation(
    problem,
    config=dict(
        optimize=dict(n_starts=50, method="Nelder-Mead"),
        profile=dict(method="Nelder-Mead"),
        sample=dict(n_samples=10_000, n_chains=3),
    ),
    overwrite=False
)

simulate_experiment(
    t_eval=np.linspace(0, 1, 100),
    conds={"fA0": 0.30, "cM0": 2.0},
    noise_level=0.01
)


simulate_experiments(
    exp_id=["exp1", "exp2"]
    t_eval=[np.linspace(0, 1, 10)] * 2
    conds=dict(
        fA0=[0.30, 0.50],
        cM0=[1.0, 2.0],
    ),
    true_params=ParameterSet.from_dict({"rA": 0.1, "rB": 0.2})
    noise_level=0.02
)


simulate_experiments(
    model=model,
    sim_params=sim_params,
    conds=dict(fA0=[0.30, 0.50], cM0=[1.0, 2.0]),
    t_eval=[np.linspace(0, 1, 10)] * 2,
    noise_level=0.01
)

"""
# Problem is a collection of experiments?
# Study is a collection of problems?
# Loop through all conditions, all parameter values
