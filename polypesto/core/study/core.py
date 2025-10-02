from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, TypeAlias, Dict, List, Any
from pathlib import Path

from polypesto.utils import filepath, read_json
from ..pypesto import Result
from ..problem.simulate import SimulatedProblem


class StudyKey(str):
    def __new__(cls, prob_id: str, param_id: str):
        return str.__new__(cls, f"{prob_id} | {param_id}")

    @property
    def prob_id(self) -> str:
        return self.split(" | ")[0]

    @property
    def param_id(self) -> str:
        return self.split(" | ")[1]

    @classmethod
    def from_string(cls, key_str: str) -> StudyKey:
        prob_id, param_id = key_str.split(" | ")
        return cls(prob_id, param_id)


# Define generic dictionary for {(prob_id, param_id): T}
T = TypeVar("T")
StudyDict: TypeAlias = Dict[StudyKey, T]
ResultsDict: TypeAlias = StudyDict[Result]
SimulatedProblemDict: TypeAlias = StudyDict[SimulatedProblem]


class StudyPaths:

    def __init__(self, study_dir: str | Path):
        self.study_dir = Path(study_dir)

    @filepath
    def metadata(self) -> Path:
        """Path to study metadata JSON file."""
        return self.study_dir / "metadata.json"

    @filepath
    def true_params(self) -> Path:
        """Path to all study true parameters (ParameterGroup) JSON file."""
        return self.study_dir / "true_params.json"

    def prob_dir(self, key: StudyKey) -> Path:
        return self.study_dir / key.param_id / key.prob_id


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
