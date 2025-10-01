from __future__ import annotations
from typing import TypeVar, TypeAlias, Dict, List, NamedTuple

from ..pypesto import Result
from ..problem import Problem, ProblemPaths
from ..conditions import SimConditions


class StudyKey(str):
    def __new__(cls, cond_id: str, param_id: str):
        return str.__new__(cls, f"{cond_id} | {param_id}")

    @property
    def cond_id(self) -> str:
        return self.split(" | ")[0]

    @property
    def param_id(self) -> str:
        return self.split(" | ")[1]

    @classmethod
    def from_string(cls, key_str: str) -> "StudyKey":
        cond_id, param_id = key_str.split(" | ")
        return cls(cond_id, param_id)


# Define generic dictionary for {(prob_id, param_id): T}
T = TypeVar("T")
StudyDict: TypeAlias = Dict[StudyKey, T]


ResultsDict: TypeAlias = StudyDict[Result]
ProblemDict: TypeAlias = StudyDict[Problem]
ProblemPathsDict: TypeAlias = StudyDict[ProblemPaths]
ConditionsDict: TypeAlias = StudyDict[List[SimConditions]]
