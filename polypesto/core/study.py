from typing import Type, TypeAlias, Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import os

from pypesto import Result, store

import pandas as pd
from matplotlib.axes import Axes
from polypesto.models import ModelInterface
from polypesto.core.params import ParameterGroup
from polypesto.core.experiment import (
    SimulatedExperiment,
    SimulationConditions,
    run_parameter_estimation,
    simulate_experiment,
)
from polypesto.utils.paths import find_experiment_paths

SimulatedExperimentDict: TypeAlias = Dict[Tuple[str, str], SimulatedExperiment]
ResultsDict: TypeAlias = Dict[Tuple[str, str], Result]


class Study:

    def __init__(
        self,
        model: Type[ModelInterface],
        simulation_params: ParameterGroup,
        experiments: SimulatedExperimentDict,
        results: Optional[ResultsDict] = None,
    ):
        self.model = model
        self.simulation_params = simulation_params
        self.experiments = experiments
        self.results = results if results is not None else {}

    def get_experiments(
        self, filter_cond_id: Optional[str] = None, filter_p_id: Optional[str] = None
    ) -> SimulatedExperimentDict:
        filtered_experiments: SimulatedExperimentDict = {}

        print("Filtering experiments...")
        print(f"Filter condition ID: {filter_cond_id}")
        print(f"Filter parameter ID: {filter_p_id}")

        for (cond_id, p_id), experiment in self.experiments.items():

            if (filter_cond_id is None or cond_id == filter_cond_id) and (
                filter_p_id is None or p_id == filter_p_id
            ):
                filtered_experiments[(cond_id, p_id)] = experiment
            # else:
            #     print(
            #         f"Skipping experiment {cond_id}, {p_id} as it does not match the filter."
            #     )

        return filtered_experiments

    def get_results(
        self, cond_id: Optional[str] = None, p_id: Optional[str] = None
    ) -> ResultsDict:
        filtered_results: ResultsDict = {}

        for (cond_id, p_id), result in self.results.items():
            if (cond_id is None or cond_id == cond_id) and (
                p_id is None or p_id == p_id
            ):
                filtered_results[(cond_id, p_id)] = result

        return filtered_results

    def get_parameter_ids(self) -> List[str]:
        """Get all parameter IDs from the simulation parameters."""

        keys = [p_id for (cond_id, p_id) in self.experiments.keys()]
        unique_keys = sorted(list(set(keys)))
        return unique_keys

    def get_parameter_values(self) -> Dict[str, Dict[str, float]]:
        return

    def get_condition_ids(self) -> List[str]:
        """Get all condition IDs from the simulation parameters."""

        keys = [cond_id for (cond_id, p_id) in self.experiments.keys()]
        unique_keys = sorted(list(set(keys)))
        return unique_keys

    def run_parameter_estimation(
        self, config: Dict[str, Any], overwrite: bool = False
    ) -> None:
        """Run parameter estimation for all experiments in the study."""

        from polypesto.visualization import (
            plot_results,
            plot_all_comparisons_1D,
            plot_all_comparisons_1D_fill,
        )

        print("Running parameter estimation for all experiments...")
        for (cond_id, p_id), experiment in self.experiments.items():

            result = self.results.get((cond_id, p_id), None)

            if overwrite or result is None:
                print(f"Running parameter estimation for {cond_id}, {p_id}...")
                result = run_parameter_estimation(experiment, config, result)
                self.results[(cond_id, p_id)] = result
                print("Done running parameter estimation.")
            else:
                print(f"Found existing result for {cond_id}, {p_id}.")
                print("Skipping parameter estimation.")

            print("Plotting results...")
            # plot_results(experiment, result)
            break

        print("Plotting all comparisons...")
        plot_all_comparisons_1D_fill(self)

    @staticmethod
    def load(dir_path: str, model: Type[ModelInterface]) -> "Study":

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")

        experiments = {}
        simulation_params = {}
        results = {}
        experiment_paths = find_experiment_paths(dir_path)

        if len(experiment_paths) == 0:
            raise FileNotFoundError(
                f"No experiment paths found in directory {dir_path}."
            )

        for (cond_id, p_id), paths in experiment_paths.items():

            print(f"Loading experiment {cond_id}, {p_id}...")
            experiment = SimulatedExperiment.load(paths, model)
            experiments[(cond_id, p_id)] = experiment
            simulation_params[p_id] = experiment.true_params

            if os.path.exists(paths.pypesto_results):
                result = store.read_result(paths.pypesto_results)
                results[(cond_id, p_id)] = result

        simulation_params = ParameterGroup("Loaded", simulation_params)

        print("Done loading experiments.")
        return Study(
            model=model,
            simulation_params=simulation_params,
            experiments=experiments,
            results=results,
        )


def create_study(
    model: Type[ModelInterface],
    simulation_params: ParameterGroup,
    conditions: List[SimulationConditions],
    obs_df: Optional[pd.DataFrame] = None,
    base_dir: str = "data",
    overwrite: bool = False,
) -> Study:

    # Try to load the study if it already exists (no overwrite)
    if not overwrite:
        try:
            study = Study.load(base_dir, model)
            print("Study already exists.")
            print("Loading existing study.")
            return study
        except FileNotFoundError:
            print("Study does not exist.")

    print("Creating new study.")
    experiments = {}
    for condition in conditions:
        for p_id in simulation_params.get_ids():

            experiment = simulate_experiment(
                model=model,
                true_params=simulation_params.by_id(p_id),
                conditions=condition,
                obs_df=obs_df,
                base_dir=base_dir,
            )
            experiments[(condition.name, p_id)] = experiment

    return Study(
        model=model,
        simulation_params=simulation_params,
        experiments=experiments,
    )


def get_all_ensemble_preds(study: Study, test_exp: SimulatedExperiment):
    """
    Get all ensemble predictions for the given study and test study.
    """

    from polypesto.core.pypesto import create_ensemble, predict_with_ensemble

    ensemble_preds = {}

    for (cond_id, p_id), result in study.results.items():
        
        if cond_id == "fA0_[0.7]_cM0_[1.0]" and p_id == "gradient_lg":

            exp = study.experiments[(cond_id, p_id)]
            ensemble = create_ensemble(exp, result)
            ensemble_pred = predict_with_ensemble(ensemble, test_exp, output_type="y")
            ensemble_preds[(cond_id, p_id)] = ensemble_pred
        # break

    return ensemble_preds
