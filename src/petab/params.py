import json
from typing import Any, Dict, List, Optional

# -------------------------- #
#          CONSTANTS         #
# -------------------------- #

KEY_REQUIRED_PARAMS = "REQUIRED_PARAMS"
KEY_PARAMETER_SETS = "PARAMETER_SETS"
KEY_PARAMETER_SETS_ALT = "PARAMETER_SETS:"
KEY_PARAMETERS = "PARAMETERS"


# -------------------------- #
#        DATA CLASSES        #
# -------------------------- #


class ParameterSet:
    """
    A container class for holding parameter data:
     - required_params: A list of required parameter keys.
     - parameter_sets: A dict of set_name -> list of parameter IDs.
     - parameters: A dict of parameter_id -> dict of key:value pairs.
    """

    def __init__(
        self,
        required_params: List[str],
        parameter_sets: Dict[str, List[str]],
        parameters: Dict[str, Dict[str, Any]],
        filepath: Optional[str] = None,
    ):
        self.required_params = required_params
        self.parameter_sets = parameter_sets
        self.parameters = parameters
        self.filepath = filepath

    @staticmethod
    def from_json(filepath: str) -> "ParameterSet":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ParameterSetParser._parse_data(data, filepath=filepath)
    
    # @staticmethod
    # def param_to_json(param: Dict[str, Any], dir: str) -> None:
    #     with open(dir + '/' + param['id'] + '.json', 'w') as f:
    #         json.dump(param, f, indent=4)

    def get_filepath(self) -> Optional[str]:
        """
        Returns the filepath if a file was used to create this ParameterSet.
        """
        return self.filepath

    def get_required_params(self) -> List[str]:
        """
        Returns the list of required parameters (e.g. ["rA", "rB", "rX", ...])
        """
        return self.required_params

    def get_parameter_set_names(self) -> List[str]:
        """
        Returns the list of parameter set names (e.g. ["IRREVERSIBLE", "REVERSIBLE"]).
        """
        return list(self.parameter_sets.keys())

    def get_parameter_ids_in_set(self, set_name: str) -> List[str]:
        """
        Returns the parameter IDs under a given set name.
        """
        return self.parameter_sets.get(set_name, [])
    
    def get_parameters_in_set(self, set_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Returns the parameter data under a given set name.
        """
        return {pid: self.parameters[pid] for pid in self.get_parameter_ids_in_set(set_name)}

    def get_parameter_data(self, parameter_id: str) -> Dict[str, Any]:
        """
        Returns the dictionary of parameter data for the requested parameter ID.
        If the parameter doesn't exist, returns an empty dict.
        """
        return self.parameters.get(parameter_id, {})

    def __repr__(self) -> str:
        return (
            f"<ParameterSet("
            f"required_params={self.required_params}, "
            f"parameter_sets={list(self.parameter_sets.keys())}, "
            f"parameter_set_vals={list(self.parameter_sets.values())}, "
            f"parameters={list(self.parameters.keys())}"
            f")>"
        )


# -------------------------- #
#        PARSER CLASS        #
# -------------------------- #


class ParameterSetParser:
    """
    Handles reading and validating JSON data, then creating a ParameterSet object.
    """

    @classmethod
    def _parse_data(cls, data: Dict[str, Any], filepath: Optional[str]) -> ParameterSet:
        """
        Internal method:
         1. Validates the top-level keys and structure.
         2. Validates that each parameter dict has all required parameters.
         3. Validates that parameter sets reference valid parameter IDs.
         4. Returns a constructed ParameterSet object.
        """
        required_params = cls._get_required_params(data)
        parameter_sets = cls._get_parameter_sets(data)
        parameters = cls._get_parameters(data, required_params)

        # Validate references in parameter_sets
        for set_name, param_list in parameter_sets.items():
            if not isinstance(param_list, list):
                raise ValueError(
                    f"Parameter set '{set_name}' must be a list, got '{type(param_list).__name__}'."
                )
            for pid in param_list:
                if pid not in parameters:
                    raise ValueError(
                        f"Parameter set '{set_name}' references an undefined parameter ID '{pid}'."
                    )

        return ParameterSet(
            required_params=required_params,
            parameter_sets=parameter_sets,
            parameters=parameters,
            filepath=filepath,
        )

    # -------------------------- #
    #      VALIDATION HELPERS    #
    # -------------------------- #

    @staticmethod
    def _get_required_params(data: Dict[str, Any]) -> List[str]:
        """
        Extracts and validates REQUIRED_PARAMS.
        """
        if KEY_REQUIRED_PARAMS not in data:
            raise ValueError(f"Missing top-level key '{KEY_REQUIRED_PARAMS}'.")
        required_params = data[KEY_REQUIRED_PARAMS]
        if not isinstance(required_params, list):
            raise ValueError(f"'{KEY_REQUIRED_PARAMS}' must be a list.")
        return required_params

    @staticmethod
    def _get_parameter_sets(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extracts and validates PARAMETER_SETS (or PARAMETER_SETS:).
        """
        # We check both "PARAMETER_SETS" and "PARAMETER_SETS:"
        if KEY_PARAMETER_SETS_ALT in data:
            parameter_sets = data[KEY_PARAMETER_SETS_ALT]
        else:
            parameter_sets = data.get(KEY_PARAMETER_SETS)
            if parameter_sets is None:
                raise ValueError(
                    f"Missing top-level key '{KEY_PARAMETER_SETS}' or '{KEY_PARAMETER_SETS_ALT}'."
                )

        if not isinstance(parameter_sets, dict):
            raise ValueError(f"'{KEY_PARAMETER_SETS}' must be a dictionary.")
        return parameter_sets

    @staticmethod
    def _get_parameters(
        data: Dict[str, Any], required_params: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts and validates PARAMETERS to ensure each parameter ID has all required fields.
        """
        if KEY_PARAMETERS not in data:
            raise ValueError(f"Missing top-level key '{KEY_PARAMETERS}'.")
        raw_params = data[KEY_PARAMETERS]
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"'{KEY_PARAMETERS}' must be a dictionary of parameter IDs."
            )

        parameters: Dict[str, Dict[str, Any]] = {}
        for param_id, param_data in raw_params.items():
            if not isinstance(param_data, dict):
                raise ValueError(
                    f"Parameter '{param_id}' data must be a dictionary of values."
                )
            # Check required param presence
            for req in required_params:
                if req not in param_data:
                    raise ValueError(
                        f"Parameter '{param_id}' is missing required field '{req}'."
                    )
            parameters[param_id] = param_data

        return parameters
