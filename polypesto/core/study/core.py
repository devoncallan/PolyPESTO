from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TypeAlias, TypeVar, Mapping

from polypesto.utils import filepath, read_json

from ..params import ParamID
from ..problem.simulate import SimulatedProblem
from ..pypesto import Result
from ..pypesto import Ensemble


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
EnsembleDict: TypeAlias = StudyDict[Ensemble]
SimulatedProblemDict: TypeAlias = StudyDict[SimulatedProblem]


def filter_study_dict(
    data: StudyDict[T], prob_id: str | None, param_id: str | None
) -> StudyDict[T]:

    filtered_dict = {}
    for key, value in data.items():
        if (prob_id is None or key.prob_id == prob_id) and (
            param_id is None or key.param_id == param_id
        ):
            filtered_dict[key] = value

    return filtered_dict


def filter_study_dict_by_values(
    data: StudyDict[T], param_values: Mapping[ParamID, float | None] | None = None
) -> StudyDict[T]:
    if param_values is None:
        return data


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

    @filepath
    def model_config(self) -> Path:
        """Path to model configuration JSON file."""
        return self.study_dir / "model_config.json"

    @property
    def logs_dir(self) -> Path:
        return self.study_dir / "logs"

    @filepath
    def model_logs(self) -> Path:
        return self.logs_dir / "model.log"

    def prob_dir(self, key: StudyKey) -> Path:
        return self.study_dir / key.param_id / key.prob_id

    def exists(self) -> bool:
        if not self.study_dir.exists():
            return False
        if not self.metadata.exists():
            return False
        if not self.true_params.exists():
            return False
        if not self.model_config.exists():
            return False
        return True


@dataclass
class StudyMetadata:

    model_name: str
    prob_ids: List[str]
    param_ids: List[str]
    keys: List[StudyKey]

    def __post_init__(self):
        """Validate metadata consistency."""

        for key in self.keys:
            if key.prob_id not in self.prob_ids:
                raise ValueError(
                    f"Inconsistent prob_id in keys: {key.prob_id} not in prob_ids."
                )
            if key.param_id not in self.param_ids:
                raise ValueError(
                    f"Inconsistent param_id in keys: {key.param_id} not in param_ids."
                )

    def to_dict(self) -> Dict[Any, Any]:
        return {
            "model_name": self.model_name,
            "prob_ids": self.prob_ids,
            "param_ids": self.param_ids,
            "keys": [str(key) for key in self.keys],
        }

    @staticmethod
    def from_dict(data: Dict[Any, Any]) -> StudyMetadata:
        # Convert string keys back to StudyKey objects
        keys = [StudyKey.from_string(key_str) for key_str in data["keys"]]
        return StudyMetadata(
            model_name=data["model_name"],
            prob_ids=data["prob_ids"],
            param_ids=data["param_ids"],
            keys=keys,
        )

    @staticmethod
    def load(filepath: str | Path) -> StudyMetadata:
        data = read_json(filepath)
        return StudyMetadata.from_dict(data)
