import streamlit as st
import numpy as np
import pandas as pd
from polypesto.app.session import Session, Keys
from pathlib import Path

from polypesto.core import Dataset, Experiment, Problem
from polypesto.core.pypesto import calculate_cis, create_ensemble, predict_with_ensemble
from polypesto.examples.base import DATA_DIR, output_dirs

# Model specific imports
from polypesto.models.binary import BinaryIrreversible
from polypesto.models.binary.utils import (
    create_ensemble_pred_problem,
    modify_experiments,
)

data = st.file_uploader("Upload CSV data files", accept_multiple_files=True, type="csv")

with st.form("run"):
    def enter_conditions(uploaded_file):
        df = pd.read_csv(uploaded_file)
        columns = df.columns.to_list()

        st.markdown("#### Select a column for each observable")
        xA_column = st.selectbox("Select xA column", options=columns, key=f'{uploaded_file.file_id}_xA')
        noise_xA = st.number_input("Enter xA noise", min_value=0.0, value=0.02, key=f'{uploaded_file.file_id}_noise_xA')
        xB_column = st.selectbox("Select xB column", options=columns, key=f'{uploaded_file.file_id}_xB')
        noise_xB = st.number_input("Enter xB noise", min_value=0.0, value=0.02, key=f'{uploaded_file.file_id}_noise_xB')

        st.markdown("#### Select conditions")
        cond_A0 = st.number_input("Enter A0", min_value=0.0, key=f'{uploaded_file.file_id}_A0')
        cond_B0 = st.number_input("Enter B0", min_value=0.0, key=f'{uploaded_file.file_id}_B0')

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
    
    submitted = st.form_submit_button("Create problem")

if not submitted:
    st.stop()

exps = modify_experiments(exps)
model = BinaryIrreversible(observables=["xA", "xB", "fA", "fB"])

problem = Problem.from_experiments(
    output_dir="app/outputs",
    model=model,
    experiments=exps,
)

result = problem.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=True,
    )

calculate_cis(result, ci_level=0.95)