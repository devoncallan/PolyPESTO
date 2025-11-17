import streamlit as st
import numpy as np
import pandas as pd
from polypesto.app.session import Session, Keys
from pathlib import Path

from polypesto.core import Dataset, Experiment, Problem, Result
from polypesto.core.problem.core import ProblemFigure
from polypesto.core.pypesto import calculate_cis, create_ensemble, predict_with_ensemble
from polypesto.examples.base import DATA_DIR, output_dirs

# Model specific imports
from polypesto.models.binary import BinaryIrreversible
from polypesto.models.binary.utils import (
    create_ensemble_pred_problem,
    modify_experiments,
)


from polypesto.app.session import Session, StateKey
from polypesto.app.components.stqdm import streamlit_tqdm
from polypesto.app.core import Keys
from polypesto.app.components.load import st_data_loader, st_petab_vis

exp = st.expander("Uploaded Data", expanded=True)
problem, result = st_data_loader(exp)

if problem is None or result is None:
    st.info("Please upload data to create a parameter estimation problem.")
    st.stop()


st_petab_vis(st, problem)

calculate_cis(result, ci_level=0.95)

st.write("#### Select Figure")
fig_types = st.multiselect(
    "Select figure type",
    options=[ft for ft in ProblemFigure],
    format_func=lambda x: x.name,
    label_visibility="collapsed",
)

for fig_type in fig_types:
    fig_path = problem.paths.get_figure_path(fig_type)
    image = fig_path.read_bytes()
    st.image(image, caption=f"{fig_path.name}")
