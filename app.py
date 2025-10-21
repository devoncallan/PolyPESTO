from pathlib import Path
from sys import path
from typing import Dict, List, Set

import streamlit as st

from polypesto.core.study import Study, StudyPaths
from polypesto.core.study.compare import StudyComparison

st.set_page_config(layout="wide")

if "loader" not in st.session_state:
    st.session_state["loader"] = True
if "data_dir" not in st.session_state:
    st.session_state["data_dir"] = None
if "study_dir_names" not in st.session_state:
    st.session_state["study_dir_names"] = []
if "study_comp" not in st.session_state:
    st.session_state["study_comp"] = None


def st_load_directory(_c: st._DeltaGenerator, data_dir_str: str):

    data_dir = Path(data_dir_str)
    if not data_dir.exists() or not data_dir.is_dir():
        st.session_state["data_dir"] = None
        st.session_state["study_dir_names"] = []
        return

    st.session_state["data_dir"] = data_dir

    study_dir_names = [
        item.name for item in sorted(data_dir.iterdir()) if item.is_dir()
    ]
    study_dir_names = [
        subdir for subdir in study_dir_names if StudyPaths(data_dir / subdir).exists()
    ]

    if study_dir_names:
        _c.info(
            f"Loaded {data_dir_str}. Found {len(study_dir_names)} study directories."
        )
        st.session_state["study_dir_names"] = study_dir_names
    else:
        _c.error("No study directories found.")
        st.session_state["study_dir_names"] = []


# @st.cache_resource
def st_load_study(dir: Path | str) -> Study:
    path = Path(dir)
    return Study.load(path)


def st_load_studies(_c: st._DeltaGenerator, study_dirs: List[Path]) -> None:

    studies = []

    if len(study_dirs) < 2:
        _c.error("Select at least two studies to compare.")
        return None

    my_bar = _c.progress(0)
    for i, dir in enumerate(study_dirs):

        my_bar.progress(
            (i + 0.1) / len(study_dirs),
            text=f"Loading study ({i+1}/{len(study_dirs)}) from {dir}",
        )
        study = st_load_study(dir)
        studies.append(study)

    my_bar.progress(1, text="All studies loaded.")
    my_bar.empty()

    comp = StudyComparison.from_studies(studies)
    st.session_state["study_comp"] = comp


def st_comparison_loader(c: st._DeltaGenerator) -> StudyComparison | None:

    c1, c2, c3 = c.columns([2, 3, 1])
    cinfo = c.container()
    c1.write("##### Studies Directory")
    data_dir_str = c2.text_input(
        "Base Directory", value="jobs/outputs/best_obs", label_visibility="collapsed"
    )
    c3.button(
        "Load",
        on_click=st_load_directory,
        args=(cinfo, data_dir_str),
        use_container_width=True,
    )

    data_dir: Path = st.session_state["data_dir"]
    study_dir_names: List[str] = st.session_state["study_dir_names"]

    if not data_dir or not study_dir_names:
        st.stop()

    study_paths = [data_dir / name for name in study_dir_names]

    c1, c2, c3 = c.columns([2, 3, 1])
    cinfo = c.container()
    c1.write("##### Select Studies")
    selected_names = c2.multiselect(
        "Select studies to compare",
        options=study_dir_names,
        label_visibility="collapsed",
    )
    study_paths = [data_dir / name for name in selected_names]
    c3.button(
        "Select All",
        on_click=st_load_studies,
        args=(cinfo, study_paths),
        use_container_width=True,
    )

    if st.session_state["study_comp"] is not None:
        c.success(f"Studies loaded from [{', '.join(p.stem for p in study_paths)}]")

    return st.session_state["study_comp"]


def st_key_selector(c: st._DeltaGenerator) -> None:

    study_comp: StudyComparison = st.session_state["study_comp"]
    if study_comp is None:
        c.warning("No studies loaded for comparison.")
        return

    keys = study_comp.get_keys()
    true_params = study_comp.get_true_params()

    crazy_param_dict = true_params.map_values()

    selections: Dict[str, Set[str]] = {}
    for param_id, val_pset_map in crazy_param_dict.items():

        vals = list(val_pset_map.keys())
        c1, c2 = c.columns([1, 4])
        c1.write(f"Parameter: **{param_id}**")
        select_val = c2.pills(
            label=f"Select values for parameter '{param_id}'",
            options=vals,
            default=vals[0],    
            selection_mode="single",
            label_visibility="collapsed",
            # format_func=lambda option: str(val_pset_map[option]),
        )
        selections[param_id] = val_pset_map[select_val]
        
    c.write("### Selected parameter values:")
    c.write(selections)
    select_set: Set[str] = set.intersection(*selections.values())
    c.write(f"### Selected parameter set IDs:")
    c.write(select_set)

    # selections = {}
    # for param_id, val_pset_map in crazy_param_dict.items():

    # values = {param_id: list(set(vals)) for param_id, vals in params_by_id.items()}

    # selections = {}
    # for param_id, pset_val_map in params_by_id.items():
    #     c1, c2 = c.columns([1, 4])
    #     c1.write(f"Parameter: **{param_id}**")
    #     selection = c2.pills(
    #         label=f"Select values for parameter '{param_id}'",
    #         options=pset_val_map.keys(),
    #         format_func=lambda option: str(pset_val_map[option]),
    #         selection_mode="single",
    #         label_visibility="collapsed",
    #     )
    #     selections[param_id] = selection

    # c.write("### Selected parameter values:")
    # c.write(selections)
    # param_key = c.select_slider(label="Select", options=list(values.keys()))

    # param_names = true_params.transpose_values()
    # # param_values_dict = {}

    # for key, value in true_params.to_dict().items():
    #     c.write(f"Key: {key}, Value: {value}")

    pass


st.title("PolyPESTO Study Comparison")

study_comp: StudyComparison = st.session_state["study_comp"]
section = st.expander("Load studies for comparison", expanded=not study_comp)
st_comparison_loader(section)

if st.session_state["study_comp"] is None:
    st.write("No studies loaded for comparison.")
    st.stop()

n_studies = len(study_comp)


st.write(f"Loaded studies: {', '.join(study_comp.keys())}")

keys = study_comp.get_keys()
# st.write(f"Loaded studies: {', '.join(keys)}")


study_key = keys[-1]
df = study_comp.get_comparison(study_key)
st.write(f"Comparison for key: {study_key}")
# st.dataframe(df)

from polypesto.core.problem.core import ProblemFigure


# Read image from from fig_paths and display in streamlit

# fig_type = st.selectbox("Select figure type", options=[ft.name for ft in ProblemFigure], index=4)
fig_type = st.selectbox(
    "Select figure type",
    options=[ft for ft in ProblemFigure],
    format_func=lambda x: x.name,
    index=4,
)
# st.write(fig_type)
fig_paths = study_comp.get_figure_paths(
    study_key, fig_type=fig_type
)

cols = st.columns(len(fig_paths))
for (study_name, fig_path), col in zip(fig_paths.items(), cols):
    col.markdown(f"### {study_name}")
    image = fig_path.read_bytes()
    col.image(image, caption=f"{study_name} - {fig_path.name}")

c = st.container()
st_key_selector(c)
