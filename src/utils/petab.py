import os
from typing import Dict, Iterable, Tuple, Sequence, List, Optional
from dataclasses import dataclass

import pandas as pd
import petab
import petab.v1.C as C


@dataclass
class PetabParameter:
    id: str
    scale: str
    bounds: Tuple[float, float]
    nominal_value: float
    estimate: bool


############################
### Define petab problem ###
############################


def define_parameters(parameters: List[PetabParameter]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                C.PARAMETER_ID: param.id,
                C.PARAMETER_SCALE: param.scale,
                C.LOWER_BOUND: param.bounds[0],
                C.UPPER_BOUND: param.bounds[1],
                C.NOMINAL_VALUE: param.nominal_value,
                C.ESTIMATE: param.estimate,
            }
            for param in parameters
        ]
    ).set_index(C.PARAMETER_ID)


def define_observables(
    observables: Dict[str, str], noise_value: float = 0.0
) -> pd.DataFrame:

    observable_ids = list(observables.keys())
    observable_formulas = list(observables.values())

    return pd.DataFrame(
        data={
            C.OBSERVABLE_ID: [f"obs_{id}" for id in observable_ids],
            C.OBSERVABLE_FORMULA: observable_formulas,
            C.NOISE_FORMULA: [noise_value] * len(observable_ids),
        }
    ).set_index(C.OBSERVABLE_ID)


def define_conditions(init_conditions: Dict[str, Sequence[float]]) -> pd.DataFrame:
    # Ensure all sequences have the same length
    lengths = [len(values) for values in init_conditions.values()]
    assert all(
        length == lengths[0] for length in lengths
    ), "All sequences must have the same length"

    # Create condition IDs
    num_conditions = lengths[0]
    condition_ids = [f"c_{i}" for i in range(num_conditions)]

    # Create the DataFrame
    conditions = pd.DataFrame(init_conditions)
    conditions[C.CONDITION_ID] = condition_ids
    conditions[C.CONDITION_NAME] = condition_ids

    # Set 'CONDITION_ID' as the index
    return conditions.set_index(C.CONDITION_ID)


# Maybe add a petab file directory
def write_petab_files(
    sbml_model_filepath: str,
    parameters_df: pd.DataFrame,
    observables_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
    petab_dir_name: Optional[str] = None,
) -> str:

    model_dir = os.path.dirname(sbml_model_filepath)
    model_filename = os.path.basename(sbml_model_filepath)

    # Define the PEtab directory
    petab_dir = os.path.join(model_dir, petab_dir_name) if petab_dir_name else model_dir
    os.makedirs(petab_dir, exist_ok=True)

    petab.v1.write_condition_df(
        conditions_df, os.path.join(petab_dir, "conditions.tsv")
    )
    petab.v1.write_measurement_df(
        measurements_df, os.path.join(petab_dir, "measurements.tsv")
    )
    petab.v1.write_observable_df(
        observables_df, os.path.join(petab_dir, "observables.tsv")
    )
    petab.v1.write_parameter_df(
        parameters_df, os.path.join(petab_dir, "parameters.tsv")
    )

    # Define PEtab configuration
    rel_model_filepath = (
        os.path.join("..", model_filename) if petab_dir_name else model_filename
    )
    yaml_config = {
        C.FORMAT_VERSION: 1,
        C.PARAMETER_FILE: "parameters.tsv",
        C.PROBLEMS: [
            {
                C.SBML_FILES: [rel_model_filepath],
                C.CONDITION_FILES: ["conditions.tsv"],
                C.MEASUREMENT_FILES: ["measurements.tsv"],
                C.OBSERVABLE_FILES: ["observables.tsv"],
            }
        ],
    }

    yaml_filepath = os.path.join(petab_dir, f"petab.yaml")
    petab.v1.yaml.write_yaml(yaml_config, yaml_filepath)

    # validate written PEtab files
    problem = petab.v1.Problem.from_yaml(yaml_filepath)
    petab.v1.lint.lint_problem(problem)

    return yaml_filepath
