from enum import Enum
from typing import Generic, TypeVar, Any, List, Dict
import streamlit as st
from dataclasses import dataclass
from pathlib import Path

from polypesto.core.study.compare import StudyComparison, StudyKey

T = TypeVar("T")
_T = TypeVar("_T")


@dataclass
class StateKey(Generic[T]):
    """A typed key for session state"""

    name: str
    default: T | None = None


class Keys:
    """Centralized typed keys"""

    DATA_DIR = StateKey[Path | None]("data_dir", default=None)
    STUDY_DIR_NAMES = StateKey[List[str]]("study_dir_names", default=[])
    STUDY_COMP = StateKey[StudyComparison | None]("study_comp", default=None)
    PARAM_SELECTIONS = StateKey[Dict[str, str]]("param_selections", default={})
    STUDY_KEY = StateKey[StudyKey | None]("study_key", default=None)

class Session:
    """A simple wrapper around streamlit's session_state."""

    @staticmethod
    def set(key: StateKey[T], value: T) -> None:
        st.session_state[key.name] = value

    @staticmethod
    def get(key: StateKey[T]) -> T | None:
        return st.session_state.get(key.name, key.default)

    @staticmethod
    def exists(key: StateKey[Any]) -> bool:
        return key.name in st.session_state

    @staticmethod
    def get_dict(key: StateKey[T], dict_key: str) -> T | None:
        return Session.get(key)[dict_key] if Session.exists(key) else None

    @staticmethod
    def set_dict(key: StateKey[T], dict_key: str, value: _T) -> None:
        if not Session.exists(key):
            Session.set(key, {})
        st.session_state[key.name][dict_key] = value

    @staticmethod
    def init(keys: StateKey[T] | List[StateKey[T]]) -> None:

        if isinstance(keys, StateKey):
            keys = [keys]

        for key in keys:
            if not Session.exists(key):
                Session.set(key, key.default)

    @staticmethod
    def init_to_value(keys: str | List[str], value: Any):

        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            if not Session.exists(key):
                Session.set(key, value)

    @staticmethod
    def init_to_none(keys: str | List[str]):
        Session.init_to_value(keys, None)

    @staticmethod
    def init_to_zero(keys: str | List[str]):
        Session.init_to_value(keys, 0)

    @staticmethod
    def is_none(key: str) -> bool:
        return Session.get(key) is None
