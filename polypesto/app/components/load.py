import streamlit as st
import pandas as pd

from pathlib import Path
from polypesto.app.components.stqdm import streamlit_tqdm
from polypesto.app.session import Session
from polypesto.core import Dataset, Experiment
from polypesto.app.core import Keys
from polypesto.core.problem.problem import Problem
from polypesto.models.binary.irreversible import BinaryIrreversible
from polypesto.models.binary.utils import modify_experiments


def st_data_loader(c: st._DeltaGenerator):
    
    data = c.file_uploader(
        "Upload CSV data files", accept_multiple_files=True, type="csv"
    )

    def enter_conditions(uploaded_file):
        df = pd.read_csv(uploaded_file)
        columns = df.columns.to_list()

        c.markdown("#### Select a column for each observable")

        c1, c2 = c.columns([2, 1])

        xA_column = c1.selectbox(
            "Select xA column", options=columns, key=f"{uploaded_file.file_id}_xA"
        )
        noise_xA = c2.number_input(
            "Enter xA noise",
            min_value=0.0,
            value=0.02,
            key=f"{uploaded_file.file_id}_noise_xA",
        )
        xB_column = c1.selectbox(
            "Select xB column", options=columns, key=f"{uploaded_file.file_id}_xB"
        )
        noise_xB = c2.number_input(
            "Enter xB noise",
            min_value=0.0,
            value=0.02,
            key=f"{uploaded_file.file_id}_noise_xB",
        )

        c.markdown("#### Select conditions")
        cond_A0 = c.number_input(
            "Enter A0", min_value=0.0, key=f"{uploaded_file.file_id}_A0"
        )
        cond_B0 = c.number_input(
            "Enter B0", min_value=0.0, key=f"{uploaded_file.file_id}_B0"
        )

        obs_map = {"xA": xA_column, "xB": xB_column}
        noise_map = {"xA": noise_xA, "xB": noise_xB}
        conds = {"A0": cond_A0, "B0": cond_B0}

        exp = Experiment.load(
            id=Path(uploaded_file.name).stem,
            conds=conds,  # Define initial conditions
            data=[  # Load conversion data and map to observables
                Dataset.load(
                    df,
                    tkey="Time[min]",
                    obs_map=obs_map,
                    noise_map=noise_map,
                )
            ],
        )

        return exp

    exps = []
    for uploaded_file in data:
        exps.append(enter_conditions(uploaded_file))

    submitted = c.button("Create problem")

    if submitted:
        exps = modify_experiments(exps)
        model = BinaryIrreversible(observables=["xA", "xB", "fA", "fB"])

        problem = Problem.from_experiments(
            output_dir="app/outputs",
            model=model,
            experiments=exps,
        )
        # with streamlit_tqdm(container=c):
        result = problem.run_parameter_estimation(
            config=dict(
                optimize=dict(
                    n_starts=50,
                    method="Nelder-Mead",
                    progress_bar=True,  # allow engine tqdm to emit updates
                ),
                sample=dict(n_samples=10000, n_chains=3),
            ),
            overwrite=True,
        )
        Session.set(Keys.PROBLEM, problem)
        Session.set(Keys.RESULT, result)

    problem = Session.get(Keys.PROBLEM)
    result = Session.get(Keys.RESULT)

    return problem, result

def st_petab_vis(c: st._DeltaGenerator, problem: Problem):

    c.markdown("### Petab Data")

    c.markdown("**Observables:**")
    # obs_ids = problem.petab_problem.observable_df["observableId"].to_list()
    obs_names = problem.petab_problem.observable_df["observableName"].to_list()
    obs_ids = problem.petab_problem.observable_df.index.to_list()
    c.write(str(obs_names))

    c.markdown("**Conditions:**")
    cond_ids = problem.petab_problem.condition_df.index.to_list()
    c.write(problem.petab_problem.condition_df[["A0", "B0"]])

    c.markdown("**Measurements:**")
    selected_obs_ids = c.multiselect("Filter observables", options=obs_ids, default=obs_ids, key="meas_obs_filter")
    selected_cond_ids = c.multiselect("Filter conditions", options=cond_ids, default=cond_ids, key="meas_cond_filter")
    meas_df = problem.petab_problem.measurement_df[["simulationConditionId", "observableId", "time", "measurement", "noiseParameters"]]
    # meas_df = problem.petab_problem.measurement_df
    meas_df_filtered = meas_df[meas_df["observableId"].isin(selected_obs_ids)]
    c.write(meas_df_filtered)

    # c.write(problem.petab_problem.parameter_df)
    # c.write(f"**Number of Experiments:** {len(problem.experiments)}")
    # c.write(f"**Number of Parameters:** {problem.pypesto_problem.dim_full}")
