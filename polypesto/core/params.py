from __future__ import annotations
import itertools
from typing import Dict, List, Mapping, NamedTuple, Optional, TypeAlias, Callable
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from polypesto.utils import read_json, write_json, ID

ParamID: TypeAlias = str
ParamSetID: TypeAlias = str
ParamGroupID: TypeAlias = str
DEFAULT_ID = "unnamed"


class Parameter(NamedTuple):
    """A single parameter with an ID and a value."""

    id: ParamID
    value: float


class ParameterSet(Dict[ParamID, float]):
    """A collection of parameters."""

    def __init__(self, data: Dict[ParamID, float], *, id: ParamSetID = DEFAULT_ID):
        super().__init__(data)
        self.id = id

    @staticmethod
    def empty() -> ParameterSet:
        return ParameterSet({}, id="")

    @staticmethod
    def from_dict(
        data: Dict[ParamID, float], id: ParamSetID = DEFAULT_ID
    ) -> ParameterSet:
        return ParameterSet(data, id=id)

    @staticmethod
    def load(filepath: str | Path) -> ParameterSet:
        data = read_json(filepath)
        return ParameterSet.from_dict(data)

    def write(self, filepath: str | Path) -> None:
        write_json(filepath, self.to_dict())

    def set_id(self, id: ParamSetID) -> ParameterSet:
        self.id = id
        return self

    def get_ids(self) -> List[ParamID]:
        return list(self.keys())

    def to_dict(self) -> Dict[ParamID, float]:
        return dict(self)

    def as_parameters(self) -> List[Parameter]:
        return [Parameter(id, value) for id, value in self.items()]

    @staticmethod
    def from_dict_list(
        data: Mapping[ParamID, ArrayLike], ids: Optional[List[ParamSetID]] = None
    ) -> List[ParameterSet]:

        param_ids = list(data.keys())
        param_data = {param_id: np.array(values) for param_id, values in data.items()}
        len_conds = {param_id: len(values) for param_id, values in param_data.items()}
        n_conds = len_conds[param_ids[0]]

        if not all(n == n_conds for n in len_conds.values()):
            raise ValueError(
                f"All parameter lists must have the same length. Actual lengths: {len_conds}"
            )

        ids = ids or ID.make_param_ids(n_conds)

        if len(ids) != n_conds:
            raise ValueError(
                f"Length of ids ({len(ids)}) must match number of parameter sets ({n_conds})."
            )

        return [
            ParameterSet.from_dict(
                {param_id: param_data[param_id][i] for param_id in param_ids}, id=ids[i]
            )
            for i in range(n_conds)
        ]


class ParameterGroup(Dict[ParamSetID, ParameterSet]):
    """A collection of parameter sets."""

    def __init__(
        self, data: Dict[ParamSetID, ParameterSet], *, id: ParamGroupID = DEFAULT_ID
    ):
        super().__init__(data)
        self.id = id

    @staticmethod
    def empty() -> ParameterGroup:
        return ParameterGroup({}, id="")

    def to_dict(self) -> Dict[ParamSetID, ParameterSet]:
        return dict(self)

    def write(self, filepath: str | Path) -> None:
        write_json(filepath, self.to_dict())

    def get_ids(self) -> List[ParamSetID]:
        return list(self.keys())

    @classmethod
    def from_dict(
        cls, data: Dict[ParamSetID, Dict[ParamID, float]], id: ParamGroupID = DEFAULT_ID
    ) -> ParameterGroup:

        param_sets = {
            ps_id: ParameterSet.from_dict(ps_data, id=ps_id)
            for ps_id, ps_data in data.items()
        }
        return cls(param_sets, id=id)

    @staticmethod
    def load(filepath: str | Path) -> ParameterGroup:
        data = read_json(filepath)
        return ParameterGroup.from_dict(data)

    @staticmethod
    def create_parameter_grid(
        param_ranges: Dict[ParamID, List[float]],
        id: ParamGroupID = DEFAULT_ID,
        filter_fn: Optional[Callable[[Dict[ParamID, float]], bool]] = None,
    ) -> ParameterGroup:

        pg = ParameterGroup({}, id=id)
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        num_psets = 0
        for combination in itertools.product(*param_values):

            params = dict(zip(param_names, combination))
            if filter_fn is None or filter_fn(params):

                set_id = f"p_{num_psets:03d}"
                param_set = ParameterSet.from_dict(params, id=set_id)
                pg[set_id] = param_set
                num_psets += 1

        return pg
