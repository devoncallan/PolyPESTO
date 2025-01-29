from typing import Any, Dict, List, Optional, TypeAlias
from dataclasses import dataclass, asdict
import src.utils.file as file

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

    def write(self, filepath: str, **kwargs):
        file.write_json(filepath, asdict(self))

    @staticmethod
    def load(filepath: str, **kwargs):
        data = file.read_json(filepath)
        return ParameterGroup.from_dict(data)


class ParameterContainer:

    def __init__(
        self,
        required_param_ids: List[ParameterID],
        all_parameter_sets: ParameterGroup,
        named_groups: Dict[ParameterGroupID, List[ParameterSetID]],
        filepath: Optional[str] = None,
    ):
        self.required_param_ids = required_param_ids
        self.all_parameter_sets = all_parameter_sets
        self.named_groups = named_groups
        self.filepath = filepath

    @staticmethod
    def from_json(filepath: str) -> "ParameterContainer":
        data = file.read_json(filepath, encoding="utf-8")
        return ParameterContainerParser._parse_data(data, filepath)

    def get_filepath(self) -> Optional[str]:
        return self.filepath

    def get_required_parameter_ids(self) -> List[ParameterID]:
        return self.required_param_ids

    def get_parameter_set_ids(self) -> List[ParameterSetID]:
        return self.all_parameter_sets.get_ids()

    def get_parameter_set(self, set_id: ParameterSetID) -> ParameterSet:
        return self.all_parameter_sets.by_id(set_id)

    def get_named_groups(self) -> List[ParameterGroupID]:
        return list(self.named_groups.keys())

    def get_parameter_group(self, group_id: ParameterGroupID) -> ParameterGroup:
        parameter_set_ids = self.named_groups.get(group_id, [])
        return ParameterGroup(
            id=group_id,
            parameter_sets={
                set_id: self.all_parameter_sets.by_id(set_id)
                for set_id in parameter_set_ids
            },
        )


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
