from typing import Tuple
from amici.amici import Solver, Model
from pypesto.petab import PetabImporter
from pypesto.problem import Problem as PypestoProblem
from pypesto.objective import AmiciObjective


def set_amici_solver_params(problem: PypestoProblem) -> PypestoProblem:

    solver: Solver = problem.objective.amici_solver

    solver.setAbsoluteTolerance(1e-10)
    solver.setRelativeTolerance(1e-10)
    solver.setStabilityLimitFlag(True)
    # solver.setMaxSteps(1e5)
    # solver.setLinearSolver("dense")

    problem.objective.amici_solver = solver

    return problem


def solver_settings(solver: Solver, **kwargs) -> Solver:

    solver.setNewtonMaxSteps(10000)
    solver.setNewtonDampingFactorMode(2)
    solver.setAbsoluteTolerance(1e-8)
    solver.setRelativeTolerance(1e-6)
    solver.setMaxSteps(100000)
    solver.setLinearSolver(6)

    return solver


def load_pypesto_problem(
    yaml_path: str, model_name: str, **kwargs
) -> Tuple[PetabImporter, PypestoProblem]:

    importer: PetabImporter = PetabImporter.from_yaml(
        yaml_path,
        model_name=model_name,
        base_path="",  # Use absolute paths
    )
    problem: PypestoProblem = importer.create_problem(**kwargs)
    assert isinstance(problem.objective, AmiciObjective)

    # problem.objective.amici_solver.setNewtonMaxSteps(10000)
    # problem.objective.amici_solver.setNewtonDampingFactorMode(1)
    # problem.objective.amici_solver.setAbsoluteTolerance(1e-10)
    # problem.objective.amici_solver.setRelativeTolerance(1e-6)
    # problem.objective.amici_solver.setMaxSteps(100000)
    # problem.objective.amici_solver.setMaxConvFails(100)
    # problem.objective.amici_solver.setMaxNonlinIters(10000)
    # problem.objective.amici_solver.setLinearSolver(6)
    # problem.objective.amici_solver.setStabilityLimitFlag(False)
    # problem.objective.amici_solver.setReturnDataReportingMode(0)
    # problem.objective.amici_solver.setLinearMultistepMethod(2)

    problem.objective.amici_solver.setNewtonMaxSteps(10000)
    problem.objective.amici_solver.setNewtonDampingFactorMode(2)
    problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem.objective.amici_solver.setRelativeTolerance(1e-6)
    # problem.objective.amici_solver.setStabilityLimitFlag(True)
    problem.objective.amici_solver.setMaxSteps(100000)
    problem.objective.amici_solver.setLinearSolver(6)

    return importer, problem
