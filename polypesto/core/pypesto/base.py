import pypesto
import pypesto.optimize
from pypesto.problem import Problem as PypestoProblem
from pypesto import Result


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
    import pypesto.profile as profile

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
    import pypesto.sample as sample

    sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        n_chains=n_chains,
    )

    result = sample.sample(
        problem=problem, n_samples=n_samples, sampler=sampler, **kwargs
    )

    sample.geweke_test(result)

    return result


def save_result(result: Result, filename: str, **kwargs):

    pypesto.store.write_result(
        result=result, filename=filename, overwrite=True, **kwargs
    )
