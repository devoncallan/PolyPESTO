from typing import Type, TypeAlias, Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import os

from pypesto import Result, store

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
        self, cond_id: Optional[str] = None, p_id: Optional[str] = None
    ) -> SimulatedExperimentDict:
        filtered_experiments: SimulatedExperimentDict = {}

        for (cond_id, p_id), experiment in self.experiments.items():
            if (cond_id is None or cond_id == cond_id) and (
                p_id is None or p_id == p_id
            ):
                filtered_experiments[(cond_id, p_id)] = experiment

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

    def get_condition_ids(self) -> List[str]:
        """Get all condition IDs from the simulation parameters."""

        keys = [cond_id for (cond_id, p_id) in self.experiments.keys()]
        unique_keys = sorted(list(set(keys)))
        return unique_keys

    def run_parameter_estimation(
        self, config: Dict[str, Any], overwrite: bool = False
    ) -> None:
        """Run parameter estimation for all experiments in the study."""

        for (cond_id, p_id), experiment in self.experiments.items():

            if not overwrite and (cond_id, p_id) in self.results:
                print(f"Skipping {cond_id}, {p_id} as it is already estimated.")
                continue

            print(f"Running parameter estimation for {cond_id}, {p_id}...")
            result = run_parameter_estimation(experiment, config)
            self.results[(cond_id, p_id)] = result

    @staticmethod
    def load(dir_path: str, model: Type[ModelInterface]) -> "Study":

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")

        experiments = {}
        simulation_params = {}
        results = {}
        experiment_paths = find_experiment_paths(dir_path)

        for (cond_id, p_id), paths in experiment_paths.items():

            experiment = SimulatedExperiment.load(paths, model)
            experiments[(cond_id, p_id)] = experiment
            simulation_params = experiment.true_params

            if os.path.exists(paths.pypesto_results):
                result = store.read_result(paths.pypesto_results)
                results[(cond_id, p_id)] = result

        simulation_params = ParameterGroup("Loaded", simulation_params)

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
    base_dir: str = "data",
    overwrite: bool = False,
) -> Study:

    experiments = {}
    for condition in conditions:
        for p_id in simulation_params.get_ids():

            experiment = simulate_experiment(
                model=model,
                true_params=simulation_params.by_id(p_id),
                conditions=condition,
                base_dir=base_dir,
            )
            experiments[(condition.name, p_id)] = experiment

    return Study(
        model=model,
        simulation_params=simulation_params,
        experiments=experiments,
    )
