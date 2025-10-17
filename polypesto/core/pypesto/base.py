from pathlib import Path
from typing import Callable, Optional, Tuple

import pypesto  # type: ignore
import pypesto.optimize  # type: ignore
from amici.amici import Solver, Model  # type: ignore
from pypesto import Problem as PypestoProblem  # type: ignore
from pypesto import Result
from pypesto.objective import AmiciObjective  # type: ignore
from pypesto.petab import PetabImporter  # type: ignore

from polypesto.utils import quiet


def optimize_problem(
    problem: PypestoProblem, method: str = "Nelder-Mead", **kwargs
) -> Result:
    """Run optimization to find optimal parameter values.

    Parameters
    ----------
    problem : PypestoProblem
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

    print(f"\n==== Running optimization ====")

    optimizer = pypesto.optimize.ScipyOptimizer(method=method)
    history_options = pypesto.HistoryOptions(trace_record=True)
    engine = pypesto.engine.MultiProcessEngine()

    result = pypesto.optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        engine=engine,
        history_options=history_options,
        **kwargs,
    )
    return result


def profile_problem(
    problem: PypestoProblem,
    method: str = "Nelder-Mead",
    **kwargs,
) -> Result:
    """Create profile likelihoods for parameters.

    Parameters
    ----------
    problem : PypestoProblem
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
    import pypesto.profile as profile  # type: ignore

    print(f"\n==== Running profiling ====")

    optimizer = pypesto.optimize.ScipyOptimizer(method=method)
    result = profile.parameter_profile(problem=problem, optimizer=optimizer, **kwargs)
    return result


def sample_problem(
    problem: PypestoProblem,
    n_samples: int = 10000,
    n_chains: int = 3,
    **kwargs,
) -> Result:
    """Sample from the parameter posterior distribution.

    Parameters
    ----------
    problem : PypestoProblem
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

    print(f"\n==== Running sampling ====")

    import pypesto.sample as sample  # type: ignore

    sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        n_chains=n_chains,
    )

    result = sample.sample(
        problem=problem, n_samples=n_samples, sampler=sampler, **kwargs
    )

    with quiet():
        sample.geweke_test(result)

    return result


def save_result(result: Result, filepath: str | Path, **kwargs) -> None:

    filepath = Path(filepath)
    overwrite = kwargs.pop("overwrite", False)
    if not filepath.exists() or overwrite:
        try:
            pypesto.store.write_result(result, filepath, overwrite=overwrite, **kwargs)

        except RuntimeError as e:
            print(f"Error saving results: {e}")


def load_result(filepath: str | Path, **kwargs) -> Optional[Result]:

    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        with quiet():
            return pypesto.store.read_result(filepath, **kwargs)

    except Exception as e:
        print(f"\tCould not load result from {str(filepath)}: {e}")
        return None


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
    assert isinstance(problem.objective.amici_model, Model)
    assert isinstance(problem.objective.amici_solver, Solver)

    problem.objective.amici_solver = solver_options(problem.objective.amici_solver)
    problem.objective.amici_model.amici_solver = problem.objective.amici_solver

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

    importer: PetabImporter = PetabImporter.from_yaml(yaml_path, model_name=model_name)
    problem: PypestoProblem = importer.create_problem(**kwargs)

    return importer, problem
