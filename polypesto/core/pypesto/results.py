"""
Parameter estimation result handler classes.

This module provides classes for handling different types of parameter estimation
results (optimization, profile, sampling) with a consistent interface.
"""

from typing import Optional

import numpy as np
from pypesto import Result, Problem


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


# def get_scaled_value(self, param_id: str, value):
#     """
#     Get correctly scaled parameter value based on problem definition.

#     Parameters
#     ----------
#     param_id : str
#         Parameter identifier
#     value : float
#         Original parameter value

#     Returns
#     -------
#     float
#         Parameter value in the correct scale (log10 or linear)
#     """
#     if (
#         not hasattr(self.result.problem, "x_names")
#         or param_id not in self.result.problem.x_names
#     ):
#         return value

#     idx = self.result.problem.x_names.index(param_id)
#     if (
#         hasattr(self.result.problem, "x_scales")
#         and idx < len(self.result.problem.x_scales)
#         and self.result.problem.x_scales[idx] == "log10"
#     ):
#         return np.log10(value)
#     return value


def get_true_param_values(
    result: Result, true_params: dict = {}, scaled: bool = False
) -> Result:

    if true_params == {}:
        return true_params

    problem: Problem = result.problem
    param_names = [problem.x_names[i] for i in problem.x_free_indices]
    for name in param_names:
        if name in true_params:
            value = true_params[name]
            if scaled:
                value = get_scaled_value(name, value)
            result[name] = value

    return result


class ParameterResult:
    """
    Base class for parameter estimation results with common functionality.

    This class provides a standard interface for working with parameter estimation
    results from different sources (optimization, profiling, sampling) and adding
    true parameter values for comparison.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing parameter estimation results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    parameter_indices : list of int, optional
        Indices of parameters to include in analysis/visualization
    """

    def __init__(self, result: Result, true_params=None, parameter_indices=None):
        self.result = result
        self.true_params = true_params
        problem: Problem = result.problem
        self.parameter_indices = problem.x_free_indices
        self.param_names = [problem.x_names[i] for i in self.parameter_indices]

    def get_scaled_value(self, param_id, value):
        """
        Get correctly scaled parameter value based on problem definition.

        Parameters
        ----------
        param_id : str
            Parameter identifier
        value : float
            Original parameter value

        Returns
        -------
        float
            Parameter value in the correct scale (log10 or linear)
        """
        if (
            not hasattr(self.result.problem, "x_names")
            or param_id not in self.result.problem.x_names
        ):
            return value

        idx = self.result.problem.x_names.index(param_id)
        if (
            hasattr(self.result.problem, "x_scales")
            and idx < len(self.result.problem.x_scales)
            and self.result.problem.x_scales[idx] == "log10"
        ):
            return np.log10(value)
        return value

    def get_true_parameter_values(self, scaled=True):
        """
        Get true parameter values for visualization.

        Parameters
        ----------
        scaled : bool, optional
            Whether to return values in the model's scale (log10 if applicable)

        Returns
        -------
        dict
            Dictionary mapping parameter names to their true values
        """
        if self.true_params is None:
            return {}

        result = {}
        for name in self.param_names:
            if name in self.true_params:
                value = self.true_params[name]
                if scaled:
                    value = self.get_scaled_value(name, value)
                result[name] = value

        return result


class OptimizationResult(ParameterResult):
    """
    Handle optimization-specific parameter estimation results.

    This class extends ParameterResult with functionality specific to
    optimization results, such as accessing the best parameter estimates.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing optimization results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    parameter_indices : list of int, optional
        Indices of parameters to include in analysis/visualization
    """

    def get_best_parameters(self, scaled=True):
        """
        Get best parameter values from optimization.

        Parameters
        ----------
        scaled : bool, optional
            Whether values are already in the model's scale

        Returns
        -------
        dict
            Dictionary mapping parameter names to their best estimates
        """
        if (
            not hasattr(self.result, "optimize_result")
            or self.result.optimize_result is None
        ):
            return {}

        # Get best parameter values
        best_x = self.result.optimize_result.x

        # Return as dictionary with parameter names
        result = {}
        for i, idx in enumerate(self.parameter_indices):
            if idx < len(self.param_names):
                name = self.param_names[idx]
                value = best_x[idx]
                result[name] = value

        return result

    def get_optimization_starts(self, max_starts=None):
        """
        Get all optimization start points and results.

        Parameters
        ----------
        max_starts : int, optional
            Maximum number of starts to include

        Returns
        -------
        list
            List of optimization results
        """
        if (
            not hasattr(self.result, "optimize_result")
            or self.result.optimize_result is None
        ):
            return []

        # Get all starts
        if hasattr(self.result.optimize_result, "list"):
            starts = self.result.optimize_result.list

            # Limit to max_starts if specified
            if max_starts is not None and len(starts) > max_starts:
                starts = starts[:max_starts]

            return starts

        return []


class ProfileResult(ParameterResult):
    """
    Handle profile-specific parameter estimation results.

    This class extends ParameterResult with functionality specific to
    profile results, such as accessing profile data for visualization.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing profile results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    parameter_indices : list of int, optional
        Indices of parameters to include in analysis/visualization
    profile_list_id : int, optional
        Which profile list to use if multiple are available
    """

    def __init__(
        self, result, true_params=None, parameter_indices=None, profile_list_id=0
    ):
        super().__init__(result, true_params, parameter_indices)
        self.profile_list_id = profile_list_id

    def has_profile_results(self):
        """Check if profile results are available."""
        return (
            hasattr(self.result, "profile_result")
            and self.result.profile_result is not None
        )

    def get_profile_indices(self):
        """Get indices of parameters that have profiles."""
        if not self.has_profile_results():
            return []

        # Try to get from profile result
        if (
            hasattr(self.result.profile_result, "profile_x")
            and len(self.result.profile_result.profile_x) > self.profile_list_id
        ):
            return list(
                range(len(self.result.profile_result.profile_x[self.profile_list_id]))
            )
        # Older PyPESTO versions
        elif (
            hasattr(self.result.profile_result, "x_path")
            and len(self.result.profile_result.x_path) > self.profile_list_id
        ):
            return list(
                range(len(self.result.profile_result.x_path[self.profile_list_id]))
            )

        return []


class SamplingResult(ParameterResult):
    """
    Handle sampling-specific parameter estimation results.

    This class extends ParameterResult with functionality specific to
    MCMC sampling results, such as accessing chain data for visualization.

    Parameters
    ----------
    result : pypesto.Result
        PyPESTO result object containing sampling results
    true_params : dict or ParameterSet, optional
        True parameter values for comparison
    parameter_indices : list of int, optional
        Indices of parameters to include in analysis/visualization
    """

    def has_sampling_results(self):
        """Check if sampling results are available."""
        return (
            hasattr(self.result, "sample_result")
            and self.result.sample_result is not None
        )

    def get_chain_data(self, i_chain=0, stepsize=1, burn_in=0):
        """
        Get parameter chain data for visualization.

        Parameters
        ----------
        i_chain : int, optional
            Chain index to use
        stepsize : int, optional
            Only include every nth sample
        burn_in : int or float, optional
            Number of samples to discard as burn-in,
            or fraction of chain length if < 1.0

        Returns
        -------
        dict
            Dictionary with chain data for visualization
        """
        if not self.has_sampling_results():
            return {}

        # Get trace data
        trace_x = self.result.sample_result.trace_x

        # Check if chain exists
        if i_chain >= len(trace_x):
            return {}

        # Get selected chain
        chain = trace_x[i_chain]

        # Apply burn-in
        if burn_in > 0:
            if burn_in < 1.0:  # Interpret as fraction
                burn_in = int(len(chain) * burn_in)
            chain = chain[burn_in:]

        # Apply stepsize
        chain = chain[::stepsize]

        # Extract parameters of interest
        result = {"chain": chain, "parameters": {}}

        # Extract each parameter trace
        for i, idx in enumerate(self.parameter_indices):
            name = self.param_names[i]
            values = chain[:, i]
            result["parameters"][name] = values

        return result

    def get_credible_intervals(self, alpha_levels=(0.95,), i_chain=0, burn_in=0.2):
        """
        Calculate credible intervals for parameters.

        Parameters
        ----------
        alpha_levels : tuple of float, optional
            Confidence levels to calculate
        i_chain : int, optional
            Chain index to use
        burn_in : float, optional
            Fraction of chain to discard as burn-in

        Returns
        -------
        dict
            Dictionary mapping parameter names to their credible intervals
        """
        chain_data = self.get_chain_data(i_chain=i_chain, burn_in=burn_in)
        if not chain_data:
            return {}

        result = {}
        for name, values in chain_data["parameters"].items():
            intervals = {}
            for alpha in alpha_levels:
                lower = (1.0 - alpha) / 2.0
                upper = 1.0 - lower

                # Calculate percentiles
                low_val = np.percentile(10**values, lower * 100)
                med_val = np.percentile(10**values, 50)
                up_val = np.percentile(10**values, upper * 100)

                intervals[alpha] = (low_val, med_val, up_val)

            result[name] = intervals

        return result
