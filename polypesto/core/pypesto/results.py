"""
Parameter estimation result handler classes.

This module provides classes for handling different types of parameter estimation
results (optimization, profile, sampling) with a consistent interface.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from pypesto import Result, Problem as PypestoProblem
from pypesto.sample.util import geweke_test
from petab.v1.parameters import scale, unscale


def has_optimization_results(result: Result) -> bool:
    return hasattr(result, "optimize_result") and result.optimize_result is not None


def has_profile_results(result: Result) -> bool:
    return hasattr(result, "profile_result") and result.profile_result is not None


def has_sampling_results(result: Result) -> bool:
    return hasattr(result, "sample_result") and result.sample_result is not None


def has_results(result: Optional[Result], key: Optional[str] = None) -> bool:

    if result is None:
        return False

    if key is None:
        return (
            has_optimization_results(result)
            or has_profile_results(result)
            or has_sampling_results(result)
        )
    elif key == "optimize":
        return has_optimization_results(result)
    elif key == "profile":
        return has_profile_results(result)
    elif key == "sample":
        return has_sampling_results(result)
    else:
        raise ValueError(f"Unknown result type: {key}")


def get_true_param_values(
    result: Result, true_params: Dict[str, float] = {}, scaled: bool = False
) -> Dict[str, float]:

    if true_params == {} or true_params is None:
        return {}

    problem: PypestoProblem = result.problem

    free_param_names = [problem.x_names[i] for i in problem.x_free_indices]

    params = {}
    for param_name, param_value in true_params.items():

        assert param_name in free_param_names

        if scaled:
            idx = free_param_names.index(param_name)
            param_value = scale(param_value, problem.x_scales[idx])

        params[param_name] = param_value

    return params


def get_best_optimization_params(
    result: Result, scaled: bool = True
) -> Dict[str, float]:

    if not has_optimization_results(result):
        return {}

    problem: PypestoProblem = result.problem
    best_x = result.optimize_result.x

    # Return as dictionary with parameter names
    result = {}
    for i, idx in enumerate(problem.x_free_indices):

        value = best_x[idx]
        if scaled:
            value = scale(value, problem.x_scales[idx])
        result[problem.x_names[idx]] = value

    return result


def get_chain_data(
    result: Result,
    exclude_burn_in: bool = True,
) -> Dict[str, np.ndarray]:

    if not has_sampling_results(result):
        return {}

    burn_in = 0
    if exclude_burn_in:
        # Check if burn in index is available
        if result.sample_result.burn_in is None:
            geweke_test(result)

        # Get burn in index
        burn_in = result.sample_result.burn_in

    chain = np.asarray(result.sample_result.trace_x[0, burn_in:, :])

    problem: PypestoProblem = result.problem
    param_names = [problem.x_names[i] for i in problem.x_free_indices]
    param_indices = list(range(len(param_names)))

    return {param_names[idx]: chain[:, idx] for idx in param_indices}


def calculate_cis(
    result: Result,
    ci_level: bool = 0.95,
    exclude_burn_in: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """Calculate parameter confidence intervals from sampling.

    Args:
        result (Result): Pypesto Result object with sampling results.
        ci_level (bool, optional): Confidence interval level. Defaults to 0.95.
        exclude_burn_in (bool, optional): Whether to exclude burn-in samples. Defaults to True.

    Returns:
        Dict(str, Tuple[float, float, float]): Dictionary mapping parameter names to
            (lower_bound, median, upper_bound) tuples.
    """

    if not has_sampling_results(result):
        return {}

    chain_data = get_chain_data(result, exclude_burn_in)

    if not chain_data:
        return {}

    ci_results = {}
    for name, chain in chain_data.items():

        lower = (1.0 - ci_level) / 2.0
        upper = 1.0 - lower

        lb = np.percentile(10**chain, lower * 100)
        ub = np.percentile(10**chain, upper * 100)
        median = np.percentile(10**chain, 50)
        ci_results[name] = (lb, median, ub)

    return ci_results
