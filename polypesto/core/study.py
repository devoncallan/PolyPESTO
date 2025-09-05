from typing import Type, TypeAlias, Dict, List, Optional, Tuple, Any, TypeVar, Union
from dataclasses import dataclass
import os

from pypesto import Result, store

import pandas as pd
from matplotlib.axes import Axes
from polypesto.models import ModelInterface
from polypesto.core.params import ParameterGroup
from polypesto.core.problem import (
    # SimulatedExperiment,
    # SimulationConditions,
    run_parameter_estimation,
    # simulate_experiment,
)

"""
WORK IN PROGRESS - NOT FUNCTIONAL ATM

"""

from numpy.typing import ArrayLike

from ..models import ModelBase
from .problem import Problem
from .conditions import SimConditions, create_sim_conditions
from .simulate import simulate_experiments


ProbParamKey: TypeAlias = Tuple[str, str]
T = TypeVar("T")
ProbParamDict: TypeAlias = Dict[ProbParamKey, T]

ProblemDict: TypeAlias = ProbParamDict[Problem]
ResultsDict: TypeAlias = ProbParamDict[Result]
ConditionsDict: TypeAlias = ProbParamDict[List[SimConditions]]


def _filter_dict(
    prob_param_dict: ProbParamDict[T],
    filter_prob_id: Optional[str],
    filter_p_id: Optional[str],
) -> ProbParamDict[T]:

    filtered_dict = {}
    for (prob_id, p_id), value in prob_param_dict.items():
        if (filter_prob_id is None or prob_id == filter_prob_id) and (
            filter_p_id is None or p_id == filter_p_id
        ):
            filtered_dict[(prob_id, p_id)] = value
    return filtered_dict


class Study:

    model: ModelBase
    sim_params: ParameterGroup
    problems: ProblemDict
    results: ResultsDict
    # experiments:

    def __init__(self):
        pass

    @staticmethod
    def load(data_dir: str, model: ModelBase) -> "Study":
        pass

    def get_problems(
        self, filter_prob_id: Optional[str] = None, filter_p_id: Optional[str] = None
    ) -> ProblemDict:

        return _filter_dict(self.problems, filter_prob_id, filter_p_id)

    def get_results(
        self, filter_prob_id: Optional[str] = None, filter_p_id: Optional[str] = None
    ) -> ResultsDict:

        return _filter_dict(self.results, filter_prob_id, filter_p_id)

    def get_prob_ids(self) -> List[str]:
        keys = [prob_id for prob_id, _ in self.problems.keys()]
        return sorted(list(set(keys)))

    def get_param_ids(self) -> List[str]:
        keys = [p_id for _, p_id in self.problems.keys()]
        return sorted(list(set(keys)))

    def run_parameter_estimation(
        self,
        config: Dict[str, Any],
        overwrite: bool = False,
    ) -> ResultsDict:

        for (prob_id, p_id), problem in self.problems.items():
            key: ProbParamKey = (prob_id, p_id)

            result = self.results.get(key, None)

            if overwrite or result is None:
                print(f"Running parameter estimation for {prob_id}, {p_id}...")
                result = run_parameter_estimation(problem, config, result)
                self.results[key] = result
                print("Done running parameter estimation.")
            else:
                print(f"Found existing result for {prob_id}, {p_id}.")
                print("Skipping parameter estimation.")

        return self.results


def create_study_conditions(
    true_params: ParameterGroup | Dict[str, Dict[str, float]],
    conds: Dict[str, ArrayLike] | Dict[str, List[ArrayLike]],
    t_evals: ArrayLike | List[ArrayLike] | List[List[ArrayLike]],
    noise_levels: float | List[float] = 0.0,
    prob_ids: Optional[List[str]] = None,
) -> ConditionsDict:

    if isinstance(true_params, dict):
        true_params = ParameterGroup.lazy_from_dict(true_params)
    elif not isinstance(true_params, ParameterGroup):
        raise ValueError("true_params must be a ParameterGroup or a dict.")

    sim_conds = []
    param_sets = true_params.to_dict()
    for p_id, ps in param_sets.items():

        # sim_conds.extend(
        sim_cond = create_sim_conditions(
            true_params=ps,
            conds=conds,
            t_evals=t_evals,
            noise_levels=noise_levels,
        )
        sim_conds[()]

    return sim_conds


# def create_study(
#     data_dir: str,
#     model: ModelBase,
#     simulation_params: ParameterGroup,
#     conditions: List[SimConditions],
#     base_dir: str = "data",
#     overwrite: bool = False,
# ) -> Study:

#     # Try to load the study if it already exists (no overwrite)
#     if not overwrite:
#         try:
#             study = Study.load(base_dir, model)
#             print("Study already exists.")
#             print("Loading existing study.")
#             return study
#         except FileNotFoundError:
#             print("Study does not exist.")

#     obs_df = model.get_obs_df()
#     for cond in conditions:
#         problem = simulate_experiments(data_dir=data_dir, model=model, conds=[cond])

#     print("Creating new study.")
#     experiments = {}
#     for condition in conditions:
#         for p_id in simulation_params.get_ids():

#             experiment = simulate_experiment(
#                 model=model,
#                 true_params=simulation_params.by_id(p_id),
#                 conditions=condition,
#                 obs_df=obs_df,
#                 base_dir=base_dir,
#             )
#             experiments[(condition.name, p_id)] = experiment

#     return Study(
#         model=model,
#         simulation_params=simulation_params,
#         experiments=experiments,
#     )


# class _Study:

#     def __init__(
#         self,
#         model: Type[ModelInterface],
#         simulation_params: ParameterGroup,
#         experiments: SimulatedExperimentDict,
#         results: Optional[ResultsDict] = None,
#     ):
#         self.model = model
#         self.simulation_params = simulation_params
#         self.experiments = experiments
#         self.results = results if results is not None else {}

#     def get_experiments(
#         self, filter_cond_id: Optional[str] = None, filter_p_id: Optional[str] = None
#     ) -> SimulatedExperimentDict:
#         filtered_experiments: SimulatedExperimentDict = {}

#         print("Filtering experiments...")
#         print(f"Filter condition ID: {filter_cond_id}")
#         print(f"Filter parameter ID: {filter_p_id}")

#         for (cond_id, p_id), experiment in self.experiments.items():

#             if (filter_cond_id is None or cond_id == filter_cond_id) and (
#                 filter_p_id is None or p_id == filter_p_id
#             ):
#                 filtered_experiments[(cond_id, p_id)] = experiment
#             # else:
#             #     print(
#             #         f"Skipping experiment {cond_id}, {p_id} as it does not match the filter."
#             #     )

#         return filtered_experiments

#     def get_results(
#         self, cond_id: Optional[str] = None, p_id: Optional[str] = None
#     ) -> ResultsDict:
#         filtered_results: ResultsDict = {}

#         for (cond_id, p_id), result in self.results.items():
#             if (cond_id is None or cond_id == cond_id) and (
#                 p_id is None or p_id == p_id
#             ):
#                 filtered_results[(cond_id, p_id)] = result

#         return filtered_results

#     def get_parameter_ids(self) -> List[str]:
#         """Get all parameter IDs from the simulation parameters."""

#         keys = [p_id for (cond_id, p_id) in self.experiments.keys()]
#         unique_keys = sorted(list(set(keys)))
#         return unique_keys

#     def get_parameter_values(self) -> Dict[str, Dict[str, float]]:
#         return

#     def get_condition_ids(self) -> List[str]:
#         """Get all condition IDs from the simulation parameters."""

#         keys = [cond_id for (cond_id, p_id) in self.experiments.keys()]
#         unique_keys = sorted(list(set(keys)))
#         return unique_keys

#     def run_parameter_estimation(
#         self, config: Dict[str, Any], overwrite: bool = False
#     ) -> None:
#         """Run parameter estimation for all experiments in the study."""

#         from polypesto.visualization import (
#             plot_results,
#             plot_all_comparisons_1D,
#             plot_all_comparisons_1D_fill,
#         )

#         print("Running parameter estimation for all experiments...")
#         for (cond_id, p_id), experiment in self.experiments.items():

#             result = self.results.get((cond_id, p_id), None)

#             if overwrite or result is None:
#                 print(f"Running parameter estimation for {cond_id}, {p_id}...")
#                 result = run_parameter_estimation(experiment, config, result)
#                 self.results[(cond_id, p_id)] = result
#                 print("Done running parameter estimation.")
#             else:
#                 print(f"Found existing result for {cond_id}, {p_id}.")
#                 print("Skipping parameter estimation.")

#             print("Plotting results...")
#             # plot_results(experiment, result)
#             break

#         print("Plotting all comparisons...")
#         plot_all_comparisons_1D_fill(self)

#     @staticmethod
#     def load(dir_path: str, model: Type[ModelInterface]) -> "Study":

#         if not os.path.exists(dir_path):
#             raise FileNotFoundError(f"Directory {dir_path} does not exist.")

#         experiments = {}
#         simulation_params = {}
#         results = {}
#         experiment_paths = find_experiment_paths(dir_path)

#         if len(experiment_paths) == 0:
#             raise FileNotFoundError(
#                 f"No experiment paths found in directory {dir_path}."
#             )

#         for (cond_id, p_id), paths in experiment_paths.items():

#             print(f"Loading experiment {cond_id}, {p_id}...")
#             experiment = SimulatedExperiment.load(paths, model)
#             experiments[(cond_id, p_id)] = experiment
#             simulation_params[p_id] = experiment.true_params

#             if os.path.exists(paths.pypesto_results):
#                 result = store.read_result(paths.pypesto_results)
#                 results[(cond_id, p_id)] = result

#         simulation_params = ParameterGroup("Loaded", simulation_params)

#         print("Done loading experiments.")
#         return Study(
#             model=model,
#             simulation_params=simulation_params,
#             experiments=experiments,
#             results=results,
#         )


# def _create_study(
#     model: Type[ModelInterface],
#     simulation_params: ParameterGroup,
#     conditions: List[SimulationConditions],
#     obs_df: Optional[pd.DataFrame] = None,
#     base_dir: str = "data",
#     overwrite: bool = False,
# ) -> Study:

#     # Try to load the study if it already exists (no overwrite)
#     if not overwrite:
#         try:
#             study = Study.load(base_dir, model)
#             print("Study already exists.")
#             print("Loading existing study.")
#             return study
#         except FileNotFoundError:
#             print("Study does not exist.")

#     print("Creating new study.")
#     experiments = {}
#     for condition in conditions:
#         for p_id in simulation_params.get_ids():

#             experiment = simulate_experiment(
#                 model=model,
#                 true_params=simulation_params.by_id(p_id),
#                 conditions=condition,
#                 obs_df=obs_df,
#                 base_dir=base_dir,
#             )
#             experiments[(condition.name, p_id)] = experiment

#     return Study(
#         model=model,
#         simulation_params=simulation_params,
#         experiments=experiments,
#     )


# def get_all_ensemble_preds(study: Study, test_exp: SimulatedExperiment):
#     """
#     Get all ensemble predictions for the given study and test study.
#     """

#     from polypesto.core.pypesto import create_ensemble, predict_with_ensemble

#     ensemble_preds = {}

#     for (cond_id, p_id), result in study.results.items():

#         if cond_id == "fA0_[0.7]_cM0_[1.0]" and p_id == "gradient_lg":

#             exp = study.experiments[(cond_id, p_id)]
#             ensemble = create_ensemble(exp, result)
#             ensemble_pred = predict_with_ensemble(ensemble, test_exp, output_type="y")
#             ensemble_preds[(cond_id, p_id)] = ensemble_pred
#         # break

#     return ensemble_preds
