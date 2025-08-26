from polypesto.models import ModelInterface
from polypesto.core.params import ParameterSet, Parameter, ParameterGroup

# df = pd.read_csv("path/to/data.csv")

# xA = df["Conversion A (%)"]
# xB = df["Conversion B (%)"]
# fA0 = 0.25


# convert_to_tsv(
#     model=ModelInterface,
#     # conditions=dict(fA0s=df["fA0s"].tolist(), cM0s=df["cM0s"].tolist()),
#     observables=dict(fA=df["Conversion A"], )
# )


# def convert_to_tsv(model: ModelInterface, fA0: float, )

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


def create_dataset(data: Dict[str, pd.DataFrame]):
    pass


a_conv_test = pd.read_csv("path/to/A_Conversion.csv")
b_conv_test = pd.read_csv("path/to/B_Conversion.csv")


from dataclasses import dataclass


@dataclass
class Dataset:
    """Simple container for dataset metadata."""

    name: str
    data: pd.DataFrame
    xkey: str
    ykeys: Dict[str, str]
    cond_id: str


init_conds = ParameterGroup.lazy_from_dict(
    {
        "exp1": {"fA0": 0.25, "cM0": 1.0},
        "exp2": {"fA0": 0.3, "cM0": 1.2},
        "exp3": {"fA0": 0.35, "cM0": 1.4},
    }
)





# Mapping observables to your data column names
xkey = {"time": "Time [min]"}
ykeys = {"xA": "Conversion A", "xB": "Conversion B", "Mw": "MWD"}

xA_map = {"xA": "Conversion A"}
xB_map = {"xB": "Conversion B"}

datastes = {
    "exp1": [
        Dataset(
            name="conv_exp1",
            data=pd.read_csv("path/to/combined_dataset.csv"),
            xkey="Time [min]",
            ykeys={"xA": "Conversion A", "xB": "Conversion B"},
        )
    ],
    "exp2": [
        Dataset(
            name="conv_exp2",
            data=pd.read_csv("path/to/combined_dataset.csv"),
            xkey="Time [min]",
            ykeys={"xA": "Conversion A", "xB": "Conversion B"},
        )
    ],
}


def merge_datasets(datasets: List[Dataset]) -> Dataset:
    """Implementation later."""
    return datasets[0]


datasets = [
    Dataset(
        name="conv_exp1",
        data=pd.read_csv("path/to/combined_dataset.csv"),
        xkey="Time [min]",
        ykeys={"xA": "Conversion A", "xB": "Conversion B"},
        cond_id="exp1",
    ),
    Dataset(
        name="conv_exp2",
        data=pd.read_csv("path/to/combined_dataset.csv"),
        xkey="Time [min]",
        ykeys={"xA": "Conversion A", "xB": "Conversion B"},
        cond_id="exp2",
    ),
    merge_datasets(
        [
            Dataset(
                name="convA_exp3",
                data=pd.read_csv("path/to/convA_exp3.csv"),
                xkey="Time [min]",
                ykeys={"xA": "Conversion A"},
                cond_id="exp3",
            ),
            Dataset(
                name="convB_exp3",
                data=pd.read_csv("path/to/convB_exp3.csv"),
                xkey="Time [min]",
                ykeys={"xB": "Conversion B"},
                cond_id="exp3",
            )
        ]
    ),
]

datasets = parse_datasets(datasets, init_conds)

datasets = {
    "c0": [
        Dataset(
            name="xA_c0",
            data=pd.read_csv("path/to/dataset_1.csv"),
            xkey="Time [min]",
            ykeys={"xA": "Conversion A"},
            cond_id="c0",
        ),
        Dataset(
            name="xB_c0",
            data=pd.read_csv("path/to/dataset_2.csv"),
            xkey="Time [min]",
            ykeys={"xB": "Conversion B"},
            cond_id="c0",
        ),
        Dataset(
            name="Mw_c0",
            data=pd.read_csv("path/to/dataset_3.csv"),
            xkey="Time [min]",
            ykeys={"Mw": "MWD"},
            cond_id="c0",
        ),
    ]
}


# Or do we just do the merging beforehand, and then create a dataset?


def create_from_datasets(
    datasets: Dict[str, List[Dataset]], conds: ParameterGroup
) -> Dict[str, pd.DataFrame]:

    new_datasets = []
    for cond_id, dataset_list in datasets.items():

        assert cond_id in conds.get_ids()
        cond = conds.by_id(cond_id)

        dataset_xA = [dataset for dataset in dataset_list if dataset.ykeys.get("xA")][0]
        dataset_xB = [dataset for dataset in dataset_list if dataset.ykeys.get("xB")][0]

        xA = dataset_xA.data[dataset_xA.ykeys["xA"]].values
        xB = dataset_xB.data[dataset_xB.ykeys["xB"]].values

        fA0 = cond.by_id("fA0").value

        A = fA0 * (1 - xA)
        B = (1 - fA0) * (1 - xB)
        fA = A / (A + B)
        fB = B / (A + B)

        dataset = Dataset(
            "Composition",
            data=pd.DataFrame(
                {dataset_xA.xkey: dataset_xA.data[dataset_xA.xkey], "fA": fA, "fB": fB}
            ),
            xkey=dataset_xA.xkey,
            ykeys={"fA": "fA", "fB": "fB"},
            cond_id=cond_id,
        )

        datasets[cond_id].append(dataset)

    return datasets
    # ds_fA = Dataset(
    #     name=f"fA_{cond_id}",
    #     data=pd.DataFrame
    # )
    # fA = fA0 * (1 - xA) + (1 - fA0) * (1 - xB)
    # fB = (1 - fA0) * (1 - xA) + fA0 * (1 - xB)

    # merge

    # for i, dataset in enumerate(dataset_list):

    #     fA0 = cond.by_id("fA0").value

    # for dataset in dataset_list:
    #     # Read the dataset and set the index to the xkey
    #     df = dataset.data.set_index(dataset.xkey)
    #     # Rename ykeys to match observable names
    #     df.rename(columns=dataset.ykeys, inplace=True)
    #     # Add condition ID as a column
    #     df[C.CONDITION_ID] = dataset.cond_id

    #     # Combine with the main DataFrame
    #     combined_df = pd.concat([combined_df, df], axis=1)

    # # Store the combined DataFrame for this condition
    # datasets[cond_id] = combined_df.reset_index()


# for cond_id, dataset_list in datasets.items():

"""
Give conditions, and datasets

Combined xA and xB datasets beforehand, have everything be in one dataframe

Dataset(
    name="conv_c0",
    data=pd.read_csv("path/to/combined_dataset.csv"),
    xkey="Time [min]",
    ykeys={"xA": "Conversion A", "xB": "Conversion B"}
    cond_id="c0",
),
Dataset(
    name="mw_c0",
    data=pd.read_csv("path/to/mwd_data.csv"),
    xkey="Time [min]",
    ykeys={"Mn": "Mn (g/mol)", "Mw": "Mw (g/mol)"},
    cond_id="c0",
),


"""

datasets = [
    # A Conv
    Dataset(
        name="xA_c0",
        data=pd.read_csv("path/to/dataset_1.csv"),
        xkey="Time [min]",
        ykeys=["Conversion A"],
        cond_id="c0",
    ),
    # B Conv
    Dataset(
        name="xB_c0",
        data=pd.read_csv("path/to/dataset_2.csv"),
        xkey="Time [min]",
        ykeys=["Conversion B"],
        cond_id="c0",
    ),
    # MWD
    Dataset(
        name="Mw_c0",
        data=pd.read_csv("path/to/dataset_3.csv"),
        xkey="Time [min]",
        ykeys=["MWD"],
        cond_id="c0",
    ),
]


# Conditions: Dict[str, Dict[str, float]]
# List[Dataset]

# Merge into a single dataframe
merged_dataset = pd.concat([pd.read_csv(dataset.path) for dataset in datasets], axis=0)

# Load as conversion


from polypesto.core.experiment import Experiment, ExperimentPaths
import polypesto.core.petab as pet
from polypesto.models import ModelInterface
from polypesto.models.CRP2 import IrreversibleCPE


# def write_experiment(
#     model: ModelInterface,
#     datasets: List[Dataset],
#     init_conds: List[Dict[str, float]],
#     observables: Dict[str, str],
#     directory: str,
# ) -> ExperimentPaths:

#     cond_df = pet.define_conditions(init_conds)
#     obs_df = pet.define_observables(observables)
#     param_df = model.get_default_parameters()


#     pet.define_empty_measurements


#     # pet.define_observables(observables, noise_value=)
#     # pet.meas

# pass


def create_meas_from_datasets(
    datasets: List[Dataset],
    obs_map: Dict[str, str],
    xcol: str = "time",
) -> pd.DataFrame:

    pass


conditions_df = pet.define_conditions(init_conditions={"fA0": 0.25, "cM0": 1.0})
# IrreversibleCPE

experiment_paths = write_experiment(
    model=IrreversibleCPE,
    datasets=merged_dataset,
    init_conds=init_conds,
    observables={"A Conversion": "xA", "B Conversion": "xB"},
    directory="data_dir/",
)

# ExperimentPaths.
# Experiment.load()
