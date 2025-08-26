from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TypeAlias

import pandas as pd

from polypesto.core.params import ParameterGroup

init_conds = ParameterGroup.lazy_from_dict(
    {
        "exp1": {"fA0": 0.25, "cM0": 1.0},
        "exp2": {"fA0": 0.3, "cM0": 1.2},
        "exp3": {"fA0": 0.35, "cM0": 1.4},
    }
)

dfKey: TypeAlias = str
obsId: TypeAlias = str

# Path to data or dataframe
dataType: TypeAlias = Path | str | pd.DataFrame
# Experiment has an initial condition
# You can collect multiple datasets from a single experiment

"""

exp1 = Experiment(
    name="exp1",
    init_conds={"fA0": 0.25, "cM0": 1.2},
    data=[
        Dataset(
            path="path/to/exp1_x.csv",
            tkey="Time [min]",
            obs_map={"Conversion": "xA"},
        ),
        Dataset(
            path="path/to/dataset/exp1_Mw.csv",
            tkey="Time [min]",
            obs_map={"Conversion": "xB"},
        )
    ]
)

exp2 = Experiment(
    name="exp2",
    init_conds={"fA0": 0.3, "cM0": 1.2},
    data=[
        Dataset(
            path="path/to/exp2_x.csv",
            tkey="Time [min]",
            obs_map={"Conversion A": "xA", "Conversion B": "xB"},
        )
    ]
)
"""

@dataclass
class Dataset:
    """Container for experimental data and metadata."""
    name: str
    data: pd.DataFrame | str | Path
    tkey: str
    obs_map: Dict[dfKey, obsId]
    cond_id: str

    def __post_init__(self):

        if isinstance(self.data, (str, Path)):
            self.data = pd.read_csv(self.data)
        elif isinstance(self.data, pd.DataFrame):
            pass
        else:
            raise TypeError(
                f"Unsupported data type for Dataset.data ({type(self.data)})"
            )

        assert isinstance(self.data, pd.DataFrame)

        # E.g., dataset.obs_map = {"Conversion": "xA"}
        self.data = self.data.rename(columns=self.obs_map)
        self.data = self.data.rename(columns={self.tkey: "time"})

        # -> dataset.obs_map = {"xA": "xA"}
        self.obs_map = {v: v for v in self.obs_map.values()}
        self.tkey = "time"


def merge_datasets(datasets: List[Dataset]) -> Dataset:

    # Consider the overlap and density of data for merging
    # We want

    pass


def datasets_to_meas_df(datasets: List[Dataset]) -> pd.DataFrame:

    pass


xA_c0_data = Dataset(
    name="xA_c0",
    data=pd.read_csv("path/to/dataset_xA_c0.csv"),
    tkey="Time [min]",
    obs_map={"Conversion": "xA"},
    cond_id="exp1",
)

xB_c0_data = Dataset(
    name="xB_c0",
    data=pd.read_csv("path/to/dataset_xB_c0.csv"),
    tkey="Time [min]",
    obs_map={"Conversion": "xB"},
    cond_id="exp1",
)
x_c0_data = merge_datasets([xA_c0_data, xB_c0_data])


def process_dataset(dataset: Dataset, init_conds: ParameterGroup) -> Dataset:
    """Add fA and fB and set total conversion, x, as independent "time" variable."""

    init_cond = init_conds.by_id(dataset.cond_id)
    fA0 = init_cond.by_id("fA0")
    fB0 = 1 - fA0

    data = dataset.data.copy()
    assert "xA" in data.columns and "xB" in data.columns

    data["x"] = fA0 * data["xA"] + fB0 * data["xB"]

    mon_A = fA0 * (1 - data["xA"])
    mon_B = fB0 * (1 - data["xB"])
    data["fA"] = mon_A / (mon_A + mon_B)
    data["fB"] = mon_B / (mon_A + mon_B)

    return Dataset(
        name=dataset.name,
        data=data,
        tkey="x",
        obs_map={**dataset.obs_map, "fA": "fA", "fB": "fB"},
        cond_id=dataset.cond_id,
    )


c0_meas_df = datasets_to_meas_df([x_c0_data])


# def add_
