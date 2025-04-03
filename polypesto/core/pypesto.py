from typing import Dict, Tuple, List, Optional, Any, Union, TypeAlias
import os

import numpy as np
from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df
from amici.amici import Solver
import pypesto.optimize
from pypesto.petab import PetabImporter
from pypesto.problem import Problem
from pypesto import Result
import pypesto
import pypesto.optimize as optimize

from pypesto.ensemble import Ensemble, EnsemblePrediction
from pypesto.C import EnsembleType
from pypesto.objective import AmiciObjective
from pypesto.C import AMICI_STATUS, AMICI_T, AMICI_X, AMICI_Y
from pypesto.predict import AmiciPredictor
import pypesto.sample

from polypesto.core.experiment import Experiment


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
    optimizer = pypesto.optimize.ScipyOptimizer(method=method)
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
    optimizer = pypesto.optimize.ScipyOptimizer(method=method)
    result = pypesto.profile.parameter_profile(
        problem=problem, optimizer=optimizer, **kwargs
    )
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
    sampler = pypesto.sample.AdaptiveParallelTemperingSampler(
        internal_sampler=pypesto.sample.AdaptiveMetropolisSampler(),
        n_chains=n_chains,
    )

    result = pypesto.sample.sample(
        problem=problem, n_samples=n_samples, sampler=sampler, **kwargs
    )

    return result


def save_result(result: Result, filename: str, **kwargs):

    pypesto.store.write_result(
        result=result, filename=filename, overwrite=True, **kwargs
    )


def create_ensemble(exp: Experiment, result: Result) -> Ensemble:

    problem = exp.pypesto_problem
    x_names = problem.get_reduced_vector(problem.x_names)

    ensemble = Ensemble.from_sample(
        result=result,
        remove_burn_in=False,
        chain_slice=slice(None, None, 5),
        x_names=x_names,
        ensemble_type=EnsembleType.sample,
        lower_bound=result.problem.lb,
        upper_bound=result.problem.ub,
    )

    return ensemble


def create_predictor(exp: Experiment, output_type: str) -> AmiciPredictor:

    obj: AmiciObjective = exp.pypesto_problem.objective

    if output_type == AMICI_Y:
        output_ids = obj.amici_model.getObservableIds()
    elif output_type == AMICI_X:
        output_ids = obj.amici_model.getStateIds()
    else:
        raise ValueError(f"Unknown output type: {output_type}")

    # This post_processor will transform the output of the simulation tool
    # such that the output is compatible with the next steps.
    def post_processor(amici_outputs, output_type, output_ids):
        outputs = [
            (
                amici_output[output_type]
                if amici_output[AMICI_STATUS] == 0
                else np.full((len(amici_output[AMICI_T]), len(output_ids)), np.nan)
            )
            for amici_output in amici_outputs
        ]
        return outputs

    from functools import partial

    post_processor_bound = partial(
        post_processor,
        output_type=output_type,
        output_ids=output_ids,
    )

    predictor = AmiciPredictor(
        amici_objective=obj,
        post_processor=post_processor_bound,
        output_ids=output_ids,
    )

    return predictor


def predict_with_ensemble(
    ensemble: Ensemble,
    test_exp: Experiment,
    output_type: str = AMICI_Y,
    **kwargs,
) -> EnsemblePrediction:

    predictor = create_predictor(test_exp, output_type)

    engine = pypesto.engine.MultiProcessEngine(**kwargs)
    ensemble_pred = ensemble.predict(
        predictor=predictor,
        prediction_id=output_type,
        engine=engine,
    )
    return ensemble_pred
