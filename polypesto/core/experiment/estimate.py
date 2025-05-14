from typing import Dict, Optional, Union, Callable

from pypesto import Result, store
from polypesto.core.pypesto import (
    optimize_problem,
    profile_problem,
    sample_problem,
)

from polypesto.core.results import has_results
from . import Experiment


def run_parameter_estimation(
    exp: Experiment,
    config: dict = {},
    result: Optional[Result] = None,
    overwrite: bool = False,
    save: bool = True,
) -> Result:

    if config == {}:
        print("No parameter estimation steps configured - skipping")
        return None

    save_components: Dict[str, bool] = {"problem": True}

    def run_if_found(
        key: str, fun: Callable, result: Optional[Result] = None
    ) -> Optional[Result]:

        if key not in config:
            return result

        if not overwrite and has_results(result, key):
            print(f"\tSkipping {key} as results already exist and overwrite=False")
            return result

        print(f"\tRunning {fun.__name__} with {config[key]}")
        result = fun(exp.pypesto_problem, result=result, **config[key])
        save_components[key] = True
        return result

    result = run_if_found("optimize", optimize_problem, result=result)
    result = run_if_found("profile", profile_problem, result=result)
    result = run_if_found("sample", sample_problem, result=result)

    if result and save:
        print(f"\tSaving results to {exp.paths.pypesto_results}")

        store.write_result(
            result=result,
            filename=exp.paths.pypesto_results,
            overwrite=True,
            **save_components,
        )

    return result
