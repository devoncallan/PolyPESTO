from typing import Tuple
from amici.amici import Solver
from pypesto.petab import PetabImporter
from pypesto.problem import Problem


def set_amici_solver_params(problem: Problem) -> Problem:

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
) -> Tuple[PetabImporter, Problem]:

    importer = PetabImporter.from_yaml(
        yaml_path,
        model_name=model_name,
        base_path="",  # Use absolute paths
    )
    problem = importer.create_problem(**kwargs)

    # problem = set_amici_solver_params(problem)
    # problem.objective.amici_solver.setPreequilibration(True)
    problem.objective.amici_solver.setNewtonMaxSteps(10000)
    problem.objective.amici_solver.setNewtonDampingFactorMode(2)
    problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem.objective.amici_solver.setRelativeTolerance(1e-6)
    # problem.objective.amici_solver.setStabilityLimitFlag(True)
    problem.objective.amici_solver.setMaxSteps(100000)
    problem.objective.amici_solver.setLinearSolver(6)
    # problem.objective.amici_solver.setStabilityLimitFlag(True)
    # problem.objective.amici_solver.setLinearMultistepMethod(1)

    # print(problem.objective.amici_solver.getNewtonMaxSteps())
    # print(problem.objective.amici_solver.getNewtonDampingFactorMode())
    # print(problem.objective.amici_solver.getAbsoluteTolerance())

    return importer, problem
