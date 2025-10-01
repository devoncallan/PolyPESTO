from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, List, Dict

from .types import StudyKey


@dataclass
class StudyMetadata:

    model_name: str
    param_ids: List[str]
    cond_ids: List[str]
    problem_dirs: Dict[StudyKey, str]

    def __post_init__(self):
        """Validate metadata consistency."""
        required_keys = {
            StudyKey(cond_id, param_id)
            for cond_id in self.cond_ids
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
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[Any, Any]) -> StudyMetadata:
        return StudyMetadata(**data)
