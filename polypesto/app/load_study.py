from typing import Any, Dict, List
from pathlib import Path

import streamlit as st
import streamlit_antd_components as sac

from polypesto.core.study import Study, StudyPaths, StudyKey
from polypesto.core.study.compare import StudyComparison

from .session import Session, Keys


def st_load_directory(c: st._DeltaGenerator, data_dir_str: str):

    data_dir = Path(data_dir_str)
    if not data_dir.exists() or not data_dir.is_dir():

        Session.set(Keys.DATA_DIR, None)
        Session.set(Keys.STUDY_DIR_NAMES, [])
        return

    Session.set(Keys.DATA_DIR, data_dir)

    study_dir_names = sorted(
        [
            subdir.name
            for subdir in data_dir.iterdir()
            if Path(data_dir / subdir.name).is_dir()
            and StudyPaths(data_dir / subdir.name).exists()
        ]
    )

    if study_dir_names:
        c.info(
            f"Loaded {data_dir_str}. Found {len(study_dir_names)} study directories."
        )
        Session.set(Keys.STUDY_DIR_NAMES, study_dir_names)
    else:
        c.error("No study directories found.")
        Session.set(Keys.STUDY_DIR_NAMES, [])


def st_load_studies(c: st._DeltaGenerator, study_dirs: List[Path]) -> None:

    studies = []

    if len(study_dirs) < 2:
        c.warning("Please select at least two study directories for comparison.")
        Session.set(Keys.STUDY_COMP, None)
        return

    load_bar = c.progress(0)
    for i, dir in enumerate(study_dirs):

        load_bar.progress(
            (i + 0.1) / len(study_dirs), text=f"Loading study from {dir}..."
        )
        try:
            study = Study.load(dir)
        except Exception as e:
            c.error(f"Failed to load study from {dir}: {e}")
            Session.set(Keys.STUDY_COMP, None)
            return

        studies.append(study)
    load_bar.progress(1.0, text="All studies loaded.")
    load_bar.empty()

    study_comp = StudyComparison.from_studies(studies)
    Session.set(Keys.STUDY_COMP, study_comp)


def st_study_loader(c: st._DeltaGenerator) -> StudyComparison | None:

    c1, c2, c3 = c.columns([2, 3, 1])
    cinfo = c.container()
    c1.write("##### Studies Directory")
    data_dir_str = c2.text_input(
        "Base Directory", value="jobs/outputs/best_obs", label_visibility="collapsed"
    )
    c3.button(
        "Select",
        on_click=st_load_directory,
        args=(cinfo, data_dir_str),
        use_container_width=True,
    )

    data_dir = Session.get(Keys.DATA_DIR)
    study_dir_names = Session.get(Keys.STUDY_DIR_NAMES)

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
        "Load",
        on_click=st_load_studies,
        args=(cinfo, study_paths),
        use_container_width=True,
    )
    if Session.get(Keys.STUDY_COMP) is not None:
        c.success(f"Selected studies from [{', '.join(p.stem for p in study_paths)}].")

    return Session.get(Keys.STUDY_COMP)


def st_param_key_selector(c: st._DeltaGenerator, study_comp: StudyComparison) -> str:

    if study_comp is None:
        c.warning("No studies loaded for comparison.")
        return

    true_params = study_comp.get_true_params()
    vals = true_params.unique_values_by_param_id()

    # Render chips with availability computed from OTHER selections
    for param_id, vmap in vals.items():
        c1, c2 = c.columns([1, 9])
        c1.write(f"##### **{param_id}**")

        # Get OTHER current selections (not this param)
        other_selections = {
            k: v
            for k, v, in Session.get(Keys.PARAM_SELECTIONS).items()
            # for k, v in st.session_state["param_selections"].items()
            if k != param_id
        }

        # Compute which ids are still available given other constraints
        available_ids = set(true_params.keys())
        for other_param, other_val in other_selections.items():
            if other_val is not None:
                available_ids &= set(vals[other_param][other_val])

        # Build items with proper disabled state
        items = [sac.ChipItem(label="Any", disabled=False)]
        items = []
        for val, id_list in sorted(vmap.items()):
            # This value is available if its ids intersect with available_ids
            is_available = len(set(id_list) & available_ids) > 0
            items.append(sac.ChipItem(label=str(val), disabled=not is_available))

        # Get current selection
        current_selection = Session.get(Keys.PARAM_SELECTIONS).get(param_id, None)
        current_idx = 0
        if current_selection is not None:
            for i, item in enumerate(items[1:], 1):
                if float(item.label) == current_selection:
                    current_idx = i
                    break

        with c2:
            sel_label = sac.chip(items=items, index=current_idx, key=f"{param_id}_chip")

        # Update session state
        Session.set_dict(
            Keys.PARAM_SELECTIONS,
            param_id,
            None if sel_label == "Any" else float(sel_label),
        )

    # Compute final result
    active_values = {
        k: v for k, v in Session.get(Keys.PARAM_SELECTIONS).items() if v is not None
    }
    matched_ids = set(true_params.filter_by_values(active_values))

    if not matched_ids or len(matched_ids) != 1:
        c.error("No parameter sets match the current selection.")
    elif len(matched_ids) > 1:
        c.error("Multiple parameter sets match the current selection.")

    param_id = list(matched_ids)[0]
    return param_id


def st_key_selector(c: st._DeltaGenerator, select_dict: Dict[str, str]) -> str:

    select_key = c.selectbox(
        "Select from keys",
        options=list(select_dict.keys()),
        format_func=lambda k: f"{k}: {select_dict[k]}",
        label_visibility="collapsed",
    )
    return select_key


def st_study_key_selector(c: st._DeltaGenerator, study_comp: StudyComparison) -> None:

    c1, c2 = c.columns([1, 4])
    c1.write("##### Select Parameters")
    param_id = st_param_key_selector(c2, study_comp)

    conds = study_comp.get_conds_dict()
    conds_dict = {
        prob_id: [f"{cond.conds.to_string()}" for cond in all_conds]
        for prob_id, all_conds in conds.items()
    }
    c1, c2 = c.columns([1, 4])
    c1.write("##### Select Conditions")
    cond_id = st_key_selector(c2, conds_dict)

    key = StudyKey(cond_id, param_id)
    if key not in study_comp.get_keys():
        Session.set(Keys.STUDY_KEY, None)
        c.error(f"No study data found for key: {key}")
    
    Session.set(Keys.STUDY_KEY, key)
