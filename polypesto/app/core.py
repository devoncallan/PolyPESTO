from polypesto.core import Dataset, Experiment, Problem, Result
from polypesto.app.session import Session, StateKey


class Keys:

    PROBLEM: StateKey[Problem | None] = StateKey[Problem | None](
        "problem", default=None
    )
    RESULT: StateKey[Result | None] = StateKey[Result | None]("result", default=None)
