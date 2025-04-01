from polypesto.core.experiment.paths import ExperimentPaths
from polypesto.core.experiment.base import Experiment
from polypesto.core.experiment.simulate import (
    SimulatedExperiment,
    SimulationConditions,
    simulate_experiment,
    create_simulation_conditions
)
from polypesto.core.experiment.estimate import run_parameter_estimation


__all__ = [
    "Experiment",
    "SimulatedExperiment",
    "SimulationConditions",
    "simulate_experiment",
    "ExperimentPaths",
    "create_simulation_conditions",
    "run_parameter_estimation",
]

# ExperimentName: TypeAlias = str
# ExperimentCollection: TypeAlias = Dict[ExperimentName, Experiment]
# ResultDict: TypeAlias = Dict[ParameterSetID, Result]
# ExperimentResults: TypeAlias = Dict[ExperimentName, ResultDict]
# StepConfigDict: TypeAlias = Dict[str, Any]
# PEConfigDict: TypeAlias = Dict[str, StepConfigDict]

"""
ExperimentCollection
{
    ExperimentName:
        Experiment
}

ExperimentResults
{
    ExperimentName:
        ResultsDict
}

ResultsDict
{
    ParameterSetID:
        Result
}

"""
