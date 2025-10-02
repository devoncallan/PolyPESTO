from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Dict

from polypesto.utils import read_json
from .types import StudyKey


@dataclass
class StudyMetadata:

    model_name: str
    prob_ids: List[str]
    param_ids: List[str]

    problem_dirs: Dict[StudyKey, str]

    def __post_init__(self):
        """Validate metadata consistency."""
        required_keys = {
            StudyKey(prob_id, param_id)
            for prob_id in self.prob_ids
            for param_id in self.param_ids
        }
        actual_keys = set(self.problem_dirs.keys())
        if required_keys != actual_keys:
            raise ValueError(
                "Inconsistent problem directories. "
                f"Expected keys: {required_keys}, "
                f"but got: {actual_keys}."
            )
        self._keys = list(actual_keys)

    def get_all_keys(self) -> List[StudyKey]:
        return self._keys

    def to_dict(self) -> Dict[Any, Any]:
        return {
            "model_name": self.model_name,
            "prob_ids": self.prob_ids,
            "param_ids": self.param_ids,
            "problem_dirs": {
                str(key): value for key, value in self.problem_dirs.items()
            },
        }

    @staticmethod
    def from_dict(data: Dict[Any, Any]) -> StudyMetadata:
        # Convert string keys back to StudyKey objects
        problem_dirs = {
            StudyKey.from_string(key_str): value
            for key_str, value in data["problem_dirs"].items()
        }
        return StudyMetadata(
            model_name=data["model_name"],
            prob_ids=data["prob_ids"],
            param_ids=data["param_ids"],
            problem_dirs=problem_dirs,
        )

    @staticmethod
    def load(filepath: str | Path) -> StudyMetadata:
        data = read_json(filepath)
        return StudyMetadata.from_dict(data)
