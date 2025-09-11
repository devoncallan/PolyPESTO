from typing import Dict, Optional, Callable, Any

from pypesto import Result, store

from ..pypesto import optimize_problem, profile_problem, sample_problem
from .results import has_results
from . import Problem


def run_parameter_estimation(
    prob: Problem,
    config: Dict[str, Any] = {},
    result: Optional[Result] = None,
    save: bool = True,
    overwrite: bool = True,
) -> Result:

    if config == {}:
        print("No parameter estimation steps configured - skipping")
        return None

    save_components: Dict[str, bool] = {"problem": True}
    save_components.update({key: True for key in config.keys()})

    def run_if_found(
        key: str, fun: Callable, _result: Optional[Result] = None
    ) -> Optional[Result]:

        if key not in config:
            return _result

        if not overwrite and has_results(_result, key):
            print(f"\tUsing existing {key} results - skipping")
            return _result

        print(f"\tRunning {fun.__name__} with {config[key]}")
        _result = fun(prob.pypesto_problem, result=_result, **config[key])
        return _result

    if result is None:
        result = prob.get_results()

    result = run_if_found("optimize", optimize_problem, result)
    result = run_if_found("profile", profile_problem, result)
    result = run_if_found("sample", sample_problem, result)

    if result and save:
        print(f"\tSaving results to {prob.paths.pypesto_results}")

        try:
            store.write_result(
                result=result,
                filename=prob.paths.pypesto_results,
                overwrite=overwrite,
                **save_components,
            )
        except RuntimeError as e:
            if overwrite:
                print("Error saving results despite overwrite=True")

    return result
