import pandas as pd
from typing import List, Tuple, Dict, Optional

from polypesto.utils.process import (
    process_data,
    convert_axes,
    convert_petab,
    create_conditions,
    join_dfs,
    save_tsv,
)


def create_tsvs(
    monomer_As: List[str],
    monomer_Bs: List[str],
    condition_ids: List[str],
    fA0s: List[float],
    base_dir: str,
):
    create_measurements_dfs(monomer_As, monomer_Bs, condition_ids, fA0s, base_dir)
    create_conditions_df(condition_ids, fA0s, base_dir)


def create_measurements_dfs(
    monomer_As: List[str],
    monomer_Bs: List[str],
    condition_ids: List[str],
    fA0s: List[float],
    base_dir: str,
):
    list_measurements_dfs = []

    for i in range(len(monomer_As)):
        df = process_data(monomer_As[i], monomer_Bs[i])
        df_conv = convert_axes(df, fA0s[i])
        meas_df = convert_petab(df_conv, condition_ids[i])
        list_measurements_dfs.append(meas_df)

    joined_measurements = join_dfs(list_measurements_dfs)

    output_file = f"{base_dir}/data/petab/p_000/measurements.tsv"
    save_tsv(joined_measurements, output_file)


def create_conditions_df(condition_ids: List[str], fA0s: List[float], base_dir: str):
    list_conditions_dfs = []

    for i in range(len(condition_ids)):
        conditions_df = pd.DataFrame()
        conditions_df = create_conditions(conditions_df, condition_ids[i], fA0s[i])
        list_conditions_dfs.append(conditions_df)

    joined_conditions = join_dfs(list_conditions_dfs)

    output_file = f"{base_dir}/data/petab/common/conditions.tsv"
    save_tsv(joined_conditions, output_file)
