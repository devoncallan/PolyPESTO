"""

Compare datasets
- Same here

Handle comparison between two measurement dataframes
- Validate that the dataframes can be exactly compared

Calculating SSR

Visualize SSR

"""

from src.petab.dataset import PetabDataset
import pandas as pd
from typing import Dict
from src.utils.params import ParameterSetID


def assert_comparable(ds1: PetabDataset, ds2: PetabDataset) -> None:

    # Checks if these datasets can be compared
    dfs1 = ds1.meas_dfs
    dfs2 = ds2.meas_dfs

    if len(dfs1) != len(dfs2):
        raise Exception(
            f"Number of simulation results is different. (Length of ds1: {len(dfs1)}, Length of ds2: {len(dfs2)})"
        )

    for id, df1 in dfs1.items():
        if id not in dfs2.keys():
            raise Exception(
                f"Dataframe with ParameterSetID {id} is not present in both datasets."
            )

        df2 = dfs2[id]
        # # Check that the columns are the same
        if len(df1) != len(df2):
            raise Exception(
                f"Length of dataframes is not the same. (Length of df1: {len(df1)}, Length of df2: {len(df2)})"
            )

        # Check that data in all columns except the measurement is the same
        for col in df1.columns:

            if col not in df2.columns:
                raise Exception(f"Column {col} is not present in both dataframes.")

            if col == C.MEASUREMENT:
                continue
            if not df1[col].equals(df2[col]):
                raise Exception(
                    f"Data in column {col} is not the same between the two dataframes."
                )


def compare_datasets(ds1: PetabDataset, ds2: PetabDataset) -> PetabDataset:

    assert_comparable(ds1, ds2)

    ids = list(ds1.meas_dfs.keys())

    diff_meas_dfs = {}
    for id in ids:
        diff_df = compare_meas_dfs(ds1.meas_dfs[id], ds2.meas_dfs[id])
        diff_meas_dfs[id] = diff_df

    return PetabDataset(
        name=f"{ds1.name} - {ds2.name}",
        obs_df=ds1.obs_df,
        cond_df=ds1.cond_df,
        param_group=ds1.param_group,
        meas_dfs=diff_meas_dfs,
    )


import petab.v1.C as C


def compare_meas_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Assumes that the dataframes are comparable"""

    diff_df = df1.copy()

    residuals = df1[C.MEASUREMENT] - df2[C.MEASUREMENT]
    diff_df[C.MEASUREMENT] = residuals

    return diff_df


# ds_name = "ds_0"
# ds_dir = f"/PolyPESTO/src/data/datasets/CRP2_CPE/{ds_name}"

# # Create the model
# model = create_CRP2_CPE_Model(model_dir=ds_dir, force_compile=False)
# # model = create_CPE_Model()

# # Define a set of parameters to sweep (e.g. irreversible params, all params, base set of params, extended set, etc.)
# pc = ParameterContainer.from_json("/PolyPESTO/src/data/parameters/CRP2_CPE.json")
# # pg = pc.get_parameter_group("IRREVERSIBLE")
# pg = pc.combine_groups(["IRREVERSIBLE", "REVERSIBLE"], "ALL")

# # Define a set of conditions to generate synthetic data
# t_eval = list(np.arange(0, 1, 0.1, dtype=float))
# fA0s = np.array([0.25, 0.5, 0.75, 0.1], dtype=float)
# cM0s = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
# cond_df = create_CRP2_CPE_conditions(fA0s, cM0s)

# # Generate and save the dataset
# ds = model.generate_dataset(
#     param_group=pg, t_eval=t_eval, cond_df=cond_df, name=ds_name
# ).write(ds_dir)
