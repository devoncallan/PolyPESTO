from typing import Callable, Tuple

from amici.amici import Solver
from pypesto.petab import PetabImporter
from pypesto.problem import Problem as PypestoProblem
from pypesto.objective import AmiciObjective


def set_solver_options(
    problem: PypestoProblem, solver_options: Callable[[Solver], Solver]
) -> PypestoProblem:
    """Set solver options for a Pypesto problem.

    Args:
        problem (PypestoProblem): The Pypesto problem.
        solver_options (Callable[[Solver], Solver]): A function that takes and returns a Solver.

    Returns:
        PypestoProblem: The updated Pypesto problem.
    """

    assert isinstance(problem.objective, AmiciObjective)
    assert isinstance(problem.objective.amici_solver, Solver)

    problem.objective.amici_solver = solver_options(problem.objective.amici_solver)

    return problem


def load_pypesto_problem(
    yaml_path: str, model_name: str, **kwargs
) -> Tuple[PetabImporter, PypestoProblem]:
    """Load a PEtab problem from a YAML file.

    Args:
        yaml_path (str): Path to the PEtab YAML file.
        model_name (str): Name of the model.

    Returns:
        Tuple[PetabImporter, PypestoProblem]: The PEtab importer and the Pypesto problem.
    """

    # Use absolute paths (base_path="")
    importer: PetabImporter = PetabImporter.from_yaml(
        yaml_path, model_name=model_name, base_path=""
    )
    problem: PypestoProblem = importer.create_problem(**kwargs)

    return importer, problem
