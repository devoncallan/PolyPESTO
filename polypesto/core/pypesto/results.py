from typing import Dict, Optional, Tuple, Literal, List
from pathlib import Path

import numpy as np
import pandas as pd
from petab.v1.parameters import scale  # type: ignore
from petab.v1.parameters import unscale  # type: ignore
from pypesto import Problem as PypestoProblem  # type: ignore
from pypesto import Result
from pypesto.sample.util import geweke_test  # type: ignore


# PEOP = Literal["optimize", "profile", "sample"]

def has_problem_results(result: Result) -> bool:
    return result is not None and result.problem is not None


def has_optimization_results(result: Result) -> bool:
    return hasattr(result, "optimize_result") and len(result.optimize_result.list) > 0


def has_profile_results(result: Result) -> bool:
    return hasattr(result, "profile_result") and len(result.profile_result.list) > 0


def has_sampling_results(result: Result) -> bool:
    return hasattr(result, "sample_result") and hasattr(result.sample_result, "trace_x")


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
    result: Result, true_params: Optional[Dict[str, float]] = None, scaled: bool = False
) -> Dict[str, float]:

    if true_params is None or true_params == {}:
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
    for idx in problem.x_free_indices:

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
    ci_level: float = 0.95,
    exclude_burn_in: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """Calculate parameter confidence intervals from sampling.

    Args:
        result (Result): Pypesto Result object with sampling results.
        ci_level (float, optional): Confidence interval level. Defaults to 0.95.
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
        # lb = np.percentile(chain, lower * 100)
        # ub = np.percentile(chain, upper * 100)
        # median = np.percentile(chain, 50)
        ci_results[name] = (lb, median, ub)

    # print(f"Parameter confidence intervals ({ci_level*100:.1f}%):")
    print(f"\n==== Parameter confidence intervals ({ci_level*100:.1f}%) ====")
    for name, (lb, median, ub) in ci_results.items():
        print(f"  {name}: {median:.3f} [{lb:.3f}, {ub:.3f}]")
        

    return ci_results


def sampling_trace_dataframe(
    result: Result,
    problem: Optional[PypestoProblem] = None,
    exclude_burn_in: bool = True,
    unscale_params: bool = True,
    chain: int = 0,
    wide: bool = True,
) -> pd.DataFrame:
    """
    Convert the stored sampling trace into a DataFrame.

    If `wide=True` (default) and `chain` is set, returns one row per iteration for that
    chain with parameter columns (unscaled) and *_scaled companions, plus chain,
    iteration, neglogpost, neglogprior.

    If `wide=False`, returns the long/tidy format (one row per parameter per iteration).
    """

    if not has_sampling_results(result):
        return pd.DataFrame()

    prob: PypestoProblem = problem or result.problem
    sr = result.sample_result

    if exclude_burn_in:
        if sr.burn_in is None:
            geweke_test(result)
        burn_in = sr.burn_in or 0
    else:
        burn_in = 0

    x = sr.trace_x
    n_chain, n_iter, n_par = x.shape
    if chain < 0 or chain >= n_chain:
        raise ValueError(f"Requested chain {chain}, but only {n_chain} chains present")

    # Normalize burn-in to per-chain array
    if np.isscalar(burn_in):
        burn_in_arr = np.full(n_chain, int(burn_in))
    else:
        burn_in_arr = np.asarray(burn_in, dtype=int)
        if burn_in_arr.size != n_chain:
            raise ValueError("burn_in length does not match number of chains")

    param_names: List[str] = prob.get_reduced_vector(prob.x_names)
    param_scales: List[str] = prob.get_reduced_vector(prob.x_scales)

    neglogpost = getattr(sr, "trace_neglogpost", None)
    neglogprior = getattr(sr, "trace_neglogprior", None)

    if wide:
        rows = []
        start = min(max(burn_in_arr[chain], 0), n_iter)
        for it in range(start, n_iter):
            row = {
                "chain": chain,
                "iteration": it,
                "neglogpost": float(neglogpost[chain, it]) if neglogpost is not None else None,
                "neglogprior": float(neglogprior[chain, it]) if neglogprior is not None else None,
            }
            for p in range(n_par):
                val_scaled = float(x[chain, it, p])
                if unscale_params:
                    val = float(unscale(val_scaled, param_scales[p]))
                else:
                    val = val_scaled
                name = param_names[p]
                row[name] = val
                row[f"{name}_scaled"] = val_scaled
            rows.append(row)
        return pd.DataFrame(rows)

    # long / tidy format
    rows = []
    for chain in range(n_chain):
        start = min(max(burn_in_arr[chain], 0), n_iter)
        for it in range(start, n_iter):
            nlpost = (
                float(neglogpost[chain, it]) if neglogpost is not None else None
            )
            nlprior = (
                float(neglogprior[chain, it]) if neglogprior is not None else None
            )
            for p in range(n_par):
                val_scaled = float(x[chain, it, p])
                if unscale_params:
                    val = float(unscale(val_scaled, param_scales[p]))
                else:
                    val = val_scaled
                rows.append(
                    {
                        "chain": chain,
                        "iteration": it,
                        "parameter": param_names[p],
                        "value_scaled": val_scaled,
                        "value": val,
                        "neglogpost": nlpost,
                        "neglogprior": nlprior,
                    }
                )

    return pd.DataFrame(rows)


def save_sampling_trace(
    result: Result,
    out_path,
    problem: Optional[PypestoProblem] = None,
    overwrite: bool = False,
    exclude_burn_in: bool = True,
    unscale_params: bool = True,
    chain: int | str = 0,
    wide: bool = True,
) -> None:
    """Persist the sampling trace to CSV.

    If `chain == "all"`, concatenates all chains (wide mode loops per chain).
    """

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return

    if chain == "all":
        sr = result.sample_result
        n_chain = sr.trace_x.shape[0]
        if wide:
            dfs = []
            for ch in range(n_chain):
                dfs.append(
                    sampling_trace_dataframe(
                        result,
                        problem=problem,
                        exclude_burn_in=exclude_burn_in,
                        unscale_params=unscale_params,
                        chain=ch,
                        wide=wide,
                    )
                )
            df = pd.concat(dfs, ignore_index=True)
        else:
            # long format already iterates over all chains internally
            df = sampling_trace_dataframe(
                result,
                problem=problem,
                exclude_burn_in=exclude_burn_in,
                unscale_params=unscale_params,
                chain=0,
                wide=wide,
            )
    else:
        df = sampling_trace_dataframe(
            result,
            problem=problem,
            exclude_burn_in=exclude_burn_in,
            unscale_params=unscale_params,
            chain=chain,
            wide=wide,
        )

    if df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
