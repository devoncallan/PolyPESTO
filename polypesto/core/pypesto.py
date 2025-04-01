from typing import Dict, Tuple, List, Optional, Any, Union, TypeAlias
import os

from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df
from amici.amici import Solver
from pypesto.petab import PetabImporter
from pypesto.problem import Problem
from pypesto import Result
import pypesto
import pypesto.optimize as optimize
import pypesto.petab
import pypesto.sample as sample
import pypesto.profile as profile
import pypesto.visualize as visualize

from polypesto.core.petab import PetabData, PetabIO
from polypesto.core.params import (
    ParameterGroup,
    ParameterSet,
    ParameterID,
    ParameterSetID,
)
from polypesto.models import sbml, ModelInterface
from polypesto.core.experiment import Experiment

# from experiments.irreversible_cpe.single_rxn.experiments import ExperimentData

# Type aliases for better code clarity
# ExperimentName: TypeAlias = str
# ExperimentCollection: TypeAlias = Dict[ExperimentName, Experiment]
# ResultDict: TypeAlias = Dict[ParameterSetID, Result]
# ExperimentResults: TypeAlias = Dict[ExperimentName, ResultDict]
# StepConfigDict: TypeAlias = Dict[str, Any]
# PEConfigDict: TypeAlias = Dict[str, StepConfigDict]


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


def optimize_problem(problem: Problem, method: str = "Nelder-Mead", **kwargs) -> Result:
    """Run optimization to find optimal parameter values.

    Parameters
    ----------
    problem : Problem
        Parameter estimation problem to solve
    n_starts : int, optional
        Number of optimization starts with different initial values, by default 100
    method : str, optional
        Optimization method to use, by default "Nelder-Mead"

    Returns
    -------
    Result
        Optimization result object containing best parameters and history
    """
    optimizer = optimize.ScipyOptimizer(method=method)
    history_options = pypesto.HistoryOptions(trace_record=True)
    engine = pypesto.engine.MultiProcessEngine()

    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        engine=engine,
        history_options=history_options,
        **kwargs,
    )
    return result


def profile_problem(
    problem: Problem,
    method: str = "Nelder-Mead",
    **kwargs,
) -> Result:
    """Create profile likelihoods for parameters.

    Parameters
    ----------
    problem : Problem
        Parameter estimation problem
    method : str, optional
        Optimization method for profiling, by default "Nelder-Mead"
    result : Optional[Result], optional
        Previous optimization result to use as starting point, by default None

    Returns
    -------
    Result
        Updated result object containing parameter profiles
    """
    optimizer = optimize.ScipyOptimizer(method=method)
    result = profile.parameter_profile(problem=problem, optimizer=optimizer, **kwargs)
    return result


def sample_problem(
    problem: Problem,
    n_samples: int = 10000,
    n_chains: int = 3,
    **kwargs,
) -> Result:
    """Sample from the parameter posterior distribution.

    Parameters
    ----------
    problem : Problem
        Parameter estimation problem
    n_samples : int, optional
        Number of samples to generate, by default 10000
    n_chains : int, optional
        Number of parallel sampling chains, by default 3
    result : Optional[Result], optional
        Previous optimization result to use as starting point, by default None

    Returns
    -------
    Result
        Updated result object containing parameter samples
    """
    sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        n_chains=n_chains,
    )

    result = sample.sample(
        problem=problem, n_samples=n_samples, sampler=sampler, **kwargs
    )

    return result


def save_result(result: Result, filename: str, **kwargs):

    pypesto.store.write_result(
        result=result, filename=filename, overwrite=True, **kwargs
    )


def create_ensemble():
    pass
