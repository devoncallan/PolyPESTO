from typing import Any, Dict, List, Optional, TypeAlias
from dataclasses import dataclass, asdict
import itertools

import polypesto.utils.file as file

# -------------------------- #
#          CONSTANTS         #
# -------------------------- #

KEY_REQUIRED_PARAMS = "REQUIRED_PARAMETERS"
KEY_PARAMETER_GROUPS = "PARAMETER_GROUPS"
KEY_PARAMETER_SETS = "PARAMETER_SETS"

ParameterID: TypeAlias = str
ParameterSetID: TypeAlias = str
ParameterGroupID: TypeAlias = str


# -------------------------- #
#        DATA CLASSES        #
# -------------------------- #


@dataclass
class Parameter:
    """
    A single simulation parameter with an ID and a value.

    Example:
    ```
    p = Parameter.from_dict({
        "id": "k1",
        "value": 0.1
    })
    ```
    """

    id: ParameterID
    value: float

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Parameter":
        return Parameter(**data)

    @staticmethod
    def lazy_from_dict(
        data: Dict[str, float], id: ParameterID = "default_id"
    ) -> "Parameter":
        return Parameter(id=id, value=data[id])


@dataclass
class ParameterSet:
    """
    A collection of parameters that define a single simulation condition.

    Example:
    ```
    ps = ParameterSet.from_dict({
        "id": "slow_kinetics",
        "parameters": {
            "k1": {"id": "k1", "value": 0.1},
            "k2": {"id": "k2", "value": 0.2},
        }
    })
    ```
    """

    id: ParameterSetID
    parameters: Dict[ParameterID, Parameter]

    def __repr__(self) -> str:
        param_str = ", ".join(f"{p.id}: {p.value}" for p in self.parameters.values())
        return f"ParameterSet(id='{self.id}', parameters={{ {param_str} }})"

    @staticmethod
    def empty() -> "ParameterSet":
        return ParameterSet(id="", parameters={})

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ParameterSet":
        parameters = {
            param_id: Parameter.from_dict(param_data)
            for param_id, param_data in data["parameters"].items()
        }
        return ParameterSet(id=data["id"], parameters=parameters)

    @staticmethod
    def lazy_from_dict(
        data: Dict[ParameterID, Any], id: Optional[ParameterSetID] = "default_id"
    ) -> "ParameterSet":
        """
        ```
        ps = ParameterSet.lazy_from_dict({
            "k1": 0.1,
            "k2": 0.2,
        })
        ```

        """
        parameters = {
            param_id: Parameter(id=param_id, value=param_data)
            for param_id, param_data in data.items()
        }
        return ParameterSet(id=id, parameters=parameters)

    @staticmethod
    def load(filepath: str, **kwargs):
        data = file.read_json(filepath)
        return ParameterSet.from_dict(data)

    def write(self, filepath: str, **kwargs):
        file.write_json(filepath, asdict(self))

    def to_dict(self) -> Dict[str, float]:
        return {param.id: float(param.value) for param in self.parameters.values()}

    def by_id(self, parameter_id: ParameterID) -> Parameter:
        if parameter_id not in self.parameters:
            raise KeyError(
                f"Parameter ID '{parameter_id}' not found in ParameterSet '{self.id}'."
            )
        return self.parameters[parameter_id]

    def get_ids(self) -> List[ParameterID]:
        return list(self.parameters.keys())

    def get_parameters(self) -> List[Parameter]:
        return list(self.parameters.values())


@dataclass
class ParameterGroup:
    """
    A collection of parameter sets that define a set of simulation conditions.

    Example:
    ```
    pg = ParameterGroup.from_dict({
        "id": "irreversible kinetics",
        "parameter_sets": {
            "slow_kinetics": {
                "id": "slow_kinetics",
                "parameters": {
                    "k1": {"id": "k1", "value": 0.1},
                    "k2": {"id": "k2", "value": 0.2},
                }
            },
            "fast_kinetics: {
                "id": "fast_kinetics",
                "parameters": {
                    "k1": {"id": "k1", "value": 1.1},
                    "k2": {"id": "k2", "value": 1.6},
                }
            }
        }
    })
    ```
    """

    id: ParameterGroupID
    parameter_sets: Dict[ParameterSetID, ParameterSet]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ParameterGroup":
        parameter_sets = {
            param_set_id: ParameterSet.from_dict(param_set_data)
            for param_set_id, param_set_data in data["parameter_sets"].items()
        }
        return ParameterGroup(id=data["id"], parameter_sets=parameter_sets)

    @staticmethod
    def lazy_from_dict(
        data: Dict[str, Any], id: ParameterSetID = "default_id"
    ) -> "ParameterGroup":
        """
        ```
        pg = ParameterGroup.lazy_from_dict({
            "slow_kinetics": {
                "k1": 0.1,
                "k2": 0.2
            },
            "fast_kinetics": {
                "k1": 1.1,
                "k2": 1.6
            }
        })
        ```

        """
        parameter_sets = {
            param_set_id: ParameterSet.lazy_from_dict(param_set_data)
            for param_set_id, param_set_data in data.items()
        }
        return ParameterGroup(id=id, parameter_sets=parameter_sets)

    def to_dict(self) -> Dict[str, Any]:
        return {
            param_set.id: param_set.to_dict()
            for param_set in self.parameter_sets.values()
        }

    def add(self, parameter_set: ParameterSet):
        if not self.parameter_sets:
            self.parameter_sets = {parameter_set.id: parameter_set}
            return

        if parameter_set.id in self.parameter_sets:
            raise KeyError(
                f"ParameterSet ID '{parameter_set.id}' already exists in ParameterGroup '{self.id}'."
            )

        self.parameter_sets[parameter_set.id] = parameter_set

    def lazy_add(self, params: Dict[ParameterID, Any]):

        id = f"p_{len(self.parameter_sets):03d}"
        param_set = ParameterSet.lazy_from_dict(params, id=id)
        self.add(param_set)

    def by_id(self, parameter_set_id: ParameterSetID) -> ParameterSet:
        if parameter_set_id not in self.parameter_sets:
            raise KeyError(
                f"ParameterSet ID '{parameter_set_id}' not found in ParameterGroup '{self.id}'."
            )
        return self.parameter_sets[parameter_set_id]

    def get_ids(self) -> List[ParameterSetID]:
        return list(self.parameter_sets.keys())

    def get_parameter_sets(self) -> List[ParameterSet]:
        return list(self.parameter_sets.values())

    def write(self, filepath: str):
        file.write_json(filepath, self.to_dict())

    @staticmethod
    def load(filepath: str, **kwargs):
        data = file.read_json(filepath)
        return ParameterGroup.from_dict(data)

    @staticmethod
    def create_parameter_grid(
        parameter_ranges: Dict[ParameterID, List[float]],
        group_id: str = "parameter_grid",
        filter_fn: Optional[callable] = None,
    ) -> "ParameterGroup":
        """
        Create a parameter group from a grid of parameter values.

        Parameters
        ----------
        parameter_ranges : Dict[ParameterID, List[float]]
            Dictionary mapping parameter names to lists of values
        group_id : str, optional
            ID for the parameter group
        filter_fn : callable, optional
            Function to filter parameter combinations. Should take a dictionary
            of parameter values and return True or False.

        Returns
        -------
        ParameterGroup
            Parameter group containing all combinations of parameter values
        """
        pg = ParameterGroup(id=group_id, parameter_sets={})

        # Get parameter names and values
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        # Generate all combinations
        num_params = 0
        for combination in itertools.product(*param_values):

            param_dict = {name: value for name, value in zip(param_names, combination)}

            if filter_fn is not None and not filter_fn(param_dict):
                continue

            pg.add(ParameterSet.lazy_from_dict(param_dict, id=f"p_{num_params:03d}"))
            num_params += 1

        return pg
