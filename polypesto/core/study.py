from typing import Type, TypeAlias, Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import os

from pypesto import Result, store

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

    def get_experiment(self, cond_id: str, p_id: str) -> SimulatedExperiment:

        return self.experiments.get((cond_id, p_id))

    def run_parameter_estimation(self, config: Dict[str, Any]) -> None:
        """Run parameter estimation for all experiments in the study."""

        results = {}
        for (cond_id, p_id), experiment in self.experiments.items():

            result = run_parameter_estimation(experiment, config)
            results[(cond_id, p_id)] = result

        self.results = results

        # return

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


"""

data
    study_000
    study_001
        conditions_000
        conditions_001
            petab
                common
                p_000
                p_001

"""
