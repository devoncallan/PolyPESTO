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
    value: Any

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Parameter":
        return Parameter(**data)


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

    def to_dict(self) -> Dict[str, Any]:
        return {param.id: param.value for param in self.parameters.values()}

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

    def to_dict(self) -> dict:
        return asdict(self)

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


@dataclass
class ParameterContainer:

    required_param_ids: List[ParameterID]
    all_parameter_sets: ParameterGroup
    named_groups: Dict[ParameterGroupID, List[ParameterSetID]]
    filepath: Optional[str] = None

    @staticmethod
    def from_json(filepath: str) -> "ParameterContainer":
        data = file.read_json(filepath, encoding="utf-8")
        return ParameterContainerParser._parse_data(data, filepath)

    def add_set(self, parameter_set: ParameterSet):
        self.all_parameter_sets.add(parameter_set)

    def add_named_group(self, group_id: ParameterGroupID, group: ParameterGroup):
        if group_id in self.named_groups:
            raise KeyError(f"ParameterGroup ID '{group_id}' already exists.")

        # Get all parameters set ids in group that aren't in the all_parameter_sets
        diff = [
            set_id
            for set_id in group.get_ids()
            if set_id not in self.all_parameter_sets.get_ids()
        ]

        if len(diff) > 0:
            raise KeyError(
                f"ParameterGroup '{group_id}' references an undefined ParameterSetIDs ({diff})."
            )

        self.named_groups[group_id] = group.get_ids()

    def get_parameter_group(self, group_id: ParameterGroupID) -> ParameterGroup:
        parameter_set_ids = self.named_groups.get(group_id, [])
        if not parameter_set_ids:
            raise KeyError(f"ParameterGroup '{group_id}' not found in {self.filepath}.")
        return ParameterGroup(
            id=group_id,
            parameter_sets={
                set_id: self.all_parameter_sets.by_id(set_id)
                for set_id in parameter_set_ids
            },
        )

    def combine_groups(
        self, group_ids: List[ParameterGroupID], group_id=None
    ) -> ParameterGroup:
        parameter_sets = {}
        for id in group_ids:
            pg = self.get_parameter_group(id)
            parameter_sets.update(pg.parameter_sets)

        if group_id is None:
            group_id = "_".join(group_ids)

        pg = ParameterGroup(id=group_id, parameter_sets=parameter_sets)
        self.add_named_group(group_id, pg)

        return pg

    def write(self, filepath: str):
        out = {
            KEY_REQUIRED_PARAMS: self.required_param_ids,
            KEY_PARAMETER_GROUPS: self.named_groups,
            KEY_PARAMETER_SETS: {
                param_set.id: {
                    param.id: param.value for param in param_set.get_parameters()
                }
                for param_set in self.all_parameter_sets.get_parameter_sets()
            },
        }
        file.write_json(filepath, out)


# -------------------------- #
#        PARSER CLASS        #
# -------------------------- #


class ParameterContainerParser:
    """
    Handles reading and validating JSON data, then creating a ParameterContainer object.
    """

    @classmethod
    def _parse_data(
        cls, data: Dict[str, Any], filepath: Optional[str]
    ) -> ParameterContainer:
        """
        Internal method:
         1. Validates the top-level keys and structure.
         2. Validates that each parameter dict has all required parameters.
         3. Validates that parameter sets reference valid parameter IDs.
         4. Returns a constructed ParameterContainer object.
        """
        required_param_ids = cls._get_required_param_ids(data)
        all_parameters = cls._get_all_parameters(data, required_param_ids)
        named_groups = cls._get_named_groups(data, all_parameters.get_ids())

        return ParameterContainer(
            required_param_ids=required_param_ids,
            all_parameter_sets=all_parameters,
            named_groups=named_groups,
            filepath=filepath,
        )

    # -------------------------- #
    #      VALIDATION HELPERS    #
    # -------------------------- #

    @staticmethod
    def _get_required_param_ids(data: Dict[str, Any]) -> List[ParameterID]:
        """
        Extracts and validates REQUIRED_PARAMS.
        """
        if KEY_REQUIRED_PARAMS not in data:
            raise ValueError(f"Missing top-level key '{KEY_REQUIRED_PARAMS}'.")
        required_param_ids: List[ParameterID] = data[KEY_REQUIRED_PARAMS]
        return required_param_ids

    @staticmethod
    def _get_named_groups(
        data: Dict[str, Any],
        parameter_ids: List[ParameterID],
    ) -> Dict[ParameterGroupID, List[ParameterSetID]]:
        """
        Extracts and validates PARAMETER_GROUPS.
        """
        if KEY_PARAMETER_GROUPS not in data:
            raise ValueError(f"Missing top-level key '{KEY_PARAMETER_GROUPS}'.")

        named_groups: Dict[ParameterGroupID, List[ParameterSetID]] = data[
            KEY_PARAMETER_GROUPS
        ]

        for group_id, param_set_ids in named_groups.items():
            for param_set_id in param_set_ids:
                if param_set_id not in parameter_ids:
                    raise ValueError(
                        f"Parameter group '{group_id}' references an undefined parameter set ID '{param_set_id}'."
                    )
        return named_groups

    @staticmethod
    def _get_all_parameters(
        data: Dict[str, Any], required_param_ids: List[ParameterID]
    ) -> ParameterGroup:

        if KEY_PARAMETER_SETS not in data:
            raise ValueError(f"Missing top-level key '{KEY_PARAMETER_SETS}'.")

        RawParamSet: TypeAlias = Dict[ParameterID, Any]
        RawParamSets: TypeAlias = Dict[ParameterSetID, RawParamSet]
        raw_param_sets: RawParamSets = data[KEY_PARAMETER_SETS]

        parameter_sets: Dict[ParameterSetID, ParameterSet] = {}

        for param_set_id, raw_param_set in raw_param_sets.items():

            missing_params = [
                req for req in required_param_ids if req not in raw_param_set
            ]
            if missing_params:
                raise ValueError(
                    f"Parameter set '{param_set_id}' is missing required parameters: {missing_params}"
                )

            parameter_set = {
                param_id: Parameter(id=param_id, value=param_data)
                for param_id, param_data in raw_param_set.items()
            }

            parameter_sets[param_set_id] = ParameterSet(
                id=param_set_id, parameters=parameter_set
            )

        return ParameterGroup(id="ALL", parameter_sets=parameter_sets)
