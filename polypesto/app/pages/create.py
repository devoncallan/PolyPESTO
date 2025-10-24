import streamlit as st
import numpy as np
from polypesto.app.session import Session, Keys
from pathlib import Path

with st.form("study"):
    st.markdown("#### Enter true values for reactivity ratios ")

    rA_input = st.text_input("rA input", value="0.1, 0.5, 1.0, 2.0, 10.0")
    rB_input = st.text_input("rB input", value="0.1, 0.5, 1.0, 2.0, 10.0")

    st.markdown("#### Enter conditions")

    fA0_input = st.multiselect("fA0 input", options=[], accept_new_options=True)

    c1, c2, c3 = st.columns(3)
    start = c1.number_input("start timepoint", min_value=0.0, max_value=1.0)
    stop = c2.number_input("stop timepoint", min_value=0.0, max_value=1.0)
    num_points = c3.number_input("number of timepoints", min_value=1, max_value=100)
    noise = st.number_input("noise", min_value=0.0, max_value=1.0)

    st.markdown("#### Select model and configs")

    model_name = st.selectbox("model", options=["binary irreversible", "binary reversible"])
    observables = st.multiselect("observables", options=["FA", "FB", "fA", "fB", "xA", "xB"])

    n_starts = st.number_input("number of starts", min_value=1, max_value=200)
    n_samples = st.number_input("number of samples", min_value=1000, max_value=100000)

    output_dir = st.text_input("output directory", value="jobs/outputs")
    study_name = st.text_input("study name")

    submitted = st.form_submit_button("create study")

if not submitted:
    st.stop()

from polypesto.core import ParameterGroup
from polypesto.core.study import Study, create_study_conditions
from polypesto.models.binary import BinaryIrreversible, BinaryReversible

if model_name == "binary irreversible":
    model = BinaryIrreversible(observables=observables, obs_noise=0.01)
elif model_name == "binary reversible":
    model = BinaryReversible(observables=observables, obs_noise=0.01)
else:
    st.error("This model is not supported.")
    st.stop()

rA = list(map(float, rA_input.strip().split(",")))
rB = list(map(float, rB_input.strip().split(",")))

true_params = ParameterGroup.create_parameter_grid(
        {
            "rA": rA,
            "rB": rB,
        },
        filter_fn=lambda p: p["rB"] >= p["rA"],
    )

fA0 = []
for cond in fA0_input:
    cond_group = list(map(float, cond.strip().split(",")))
    fA0.append(cond_group)

fB0 = []
for cond in fA0:
    cond_group = []

    for val in cond:
        cond_group.append(1-val)
    
    fB0.append(cond_group)

conds_dict = create_study_conditions(
        conds=dict(
            A0=fA0,
            B0=fB0,
        ),
        t_evals=np.linspace(start, stop, num_points),
        meas_noise=noise,
    )

output_dir = Path(output_dir) / study_name

study = Study.create(
    study_dir=output_dir,
    model=model,
    true_params=true_params,
    sim_conds=conds_dict,
    overwrite=True,
)

study = Study.load(output_dir, model)

study.run_parameter_estimation(
    config=dict(
        optimize=dict(n_starts=n_starts, method="Nelder-Mead"),
        sample=dict(n_samples=n_samples, n_chains=3),
    ),
    overwrite=True,
)

st.success("success!")