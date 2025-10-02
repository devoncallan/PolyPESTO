from __future__ import annotations
from typing import TypeVar, TypeAlias, Dict

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
