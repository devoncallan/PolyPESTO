from pathlib import Path
from typing import Dict, List, Set

import streamlit as st
import streamlit_antd_components as sac

from polypesto.core.study import Study, StudyPaths
from polypesto.core.study.compare import StudyComparison
from polypesto.core.problem.core import ProblemFigure
from polypesto.app.load_study import (
    st_study_loader,
    st_param_key_selector,
    st_key_selector,
    st_study_key_selector,
)
from polypesto.app.session import Session, Keys

st.set_page_config(layout="wide")


Session.init(
    [
        Keys.DATA_DIR,
        Keys.STUDY_DIR_NAMES,
        Keys.STUDY_COMP,
        Keys.PARAM_SELECTIONS,
        Keys.STUDY_KEY,
    ]
)
st.html(Path("polypesto/app/styles.css"))
st.title("PolyPESTO Study Comparison")

# Load studies for comparison
expander_title = "Load studies for comparison"
study_comp: StudyComparison = Session.get(Keys.STUDY_COMP)
if study_comp is not None:
    expander_title += f" -------> {[name for name in study_comp.keys()]}"


section = st.expander(expander_title, expanded=not study_comp)
st_study_loader(section)

if Session.get(Keys.STUDY_COMP) is None:
    st.write("No studies loaded for comparison.")
    st.stop()

n_studies = len(study_comp)

c1, c2 = st.columns([2, 1])
st_study_key_selector(c1, study_comp)

study_key = Session.get(Keys.STUDY_KEY)
if study_key is None:
    st.write("No study key selected.")
    st.stop()

c2.write("##### Select Figure")
fig_types = c2.multiselect(
    "Select figure type",
    options=[ft for ft in ProblemFigure],
    format_func=lambda x: x.name,
    label_visibility="collapsed",
)

for fig_type in fig_types:
    fig_paths = study_comp.get_figure_paths(study_key, fig_type=fig_type)

    cols = st.columns(len(fig_paths) + 1)
    for (study_name, fig_path), col in zip(fig_paths.items(), cols):
        col.markdown(f"### {study_name}")
        image = fig_path.read_bytes()
        col.image(image, caption=f"{study_name} - {fig_path.name}")
