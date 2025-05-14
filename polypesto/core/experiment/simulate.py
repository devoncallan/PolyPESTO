from typing import Optional, Type, Dict, List, TypeAlias
from dataclasses import dataclass
import os

import pandas as pd

from polypesto.core.params import ParameterSet
from polypesto.models import ModelInterface
from polypesto.core.petab import (
    PetabData,
    define_empty_measurements,
    PetabIO,
    add_noise_to_measurements,
)

from . import Experiment, ExperimentPaths


ConditionsDict: TypeAlias = Dict[str, List[float]]


@dataclass
class SimulationConditions:
    """
    Class to hold simulation conditions for a single experiment."""

    name: str
    t_eval: List[float]
    conditions: ConditionsDict
    fit_params: Optional[pd.DataFrame]
    noise_level: float = 0.0


def create_simulation_conditions(
    conditions_dict: ConditionsDict,
) -> List[SimulationConditions]:
    """
    Create a list of SimulationConditions from a dictionary.
    Dictionary keys should be:
    - name: Name of the simulation
    - t_eval: Time points for the simulation
    - conditions: Dictionary of experimental conditions
    - fit_params: DataFrame of fit parameters
    - noise_level: Noise level for the simulation
    """

    # Check that all lists have the same length
    ntrials = len(next(iter(conditions_dict.values())))
    for key, value in conditions_dict.items():
        if key != "conditions" and len(value) != ntrials:
            print(f"Key: {key}, Value: {value}")
            raise ValueError(
                f"Length mismatch for {key}: expected {ntrials}, got {len(value)}"
            )

    conds = []
    for i in range(ntrials):

        condition_kwargs = {}
        for key, value in conditions_dict.items():
            if key == "conditions" and isinstance(value, dict):
                condition_kwargs[key] = {k: v[i] for k, v in value.items()}
            else:
                condition_kwargs[key] = value[i]

        cond = SimulationConditions(**condition_kwargs)
        conds.append(cond)

    return conds


@dataclass
class SimulatedExperiment:
    """
    Class to hold a simulated experiment.
    ----------
    experiment : Experiment
        The experiment object containing the simulated data.
    true_params : ParameterSet
        The true parameter values used for the simulation.
    """

    experiment: Experiment
    true_params: ParameterSet

    @property
    def petab_problem(self):
        return self.experiment.petab_problem

    @property
    def pypesto_problem(self):
        return self.experiment.pypesto_problem

    @property
    def paths(self):
        return self.experiment.paths

    @staticmethod
    def load(paths: ExperimentPaths, model: ModelInterface) -> "SimulatedExperiment":

        paths.assert_parameters_exist()

        experiment = Experiment.load(paths, model)
        true_params = ParameterSet.load(paths.true_params())
        return SimulatedExperiment(
            experiment=experiment,
            true_params=true_params,
        )


def create_simulated_experiment(
    model: Type[ModelInterface],
    true_params: ParameterSet,
    data: PetabData,
    paths: ExperimentPaths,
    force_compile: bool = False,
    noise_level: float = 0.0,
) -> SimulatedExperiment:

    # Write SBML model
    from polypesto.models import sbml

    sbml_filepath = sbml.write_model(
        model_def=model.sbml_model_def(), model_dir=paths.common_dir
    )

    # Write common PEtab files
    PetabIO.write_obs_df(data.obs_df, filename=paths.observables)
    PetabIO.write_cond_df(data.cond_df, filename=paths.conditions)
    PetabIO.write_param_df(data.param_df, filename=paths.fit_parameters)

    # Write experiment specific files
    true_params.write(paths.true_params())
    PetabIO.write_meas_df(data.meas_df, filename=paths.measurements())

    # Write YAML file
    PetabIO.write_yaml(
        yaml_filepath=paths.petab_yaml(),
        sbml_filepath=sbml_filepath,
        cond_filepath=paths.conditions,
        meas_filepath=paths.measurements(),
        obs_filepath=paths.observables,
        param_filepath=paths.fit_parameters,
    )

    from polypesto.core.pypesto import load_pypesto_problem
    from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df

    importer, problem = load_pypesto_problem(
        yaml_path=paths.petab_yaml(), model_name=model.name, force_compile=force_compile
    )

    # Simulate experiment
    sim_data = simulate_petab(
        petab_problem=importer.petab_problem,
        amici_model=problem.objective.amici_model,
        solver=problem.objective.amici_solver,
        problem_parameters=true_params.to_dict(),
    )

    # Create measurement DataFrame
    meas_df = rdatas_to_measurement_df(
        sim_data["rdatas"],
        problem.objective.amici_model,
        importer.petab_problem.measurement_df,
    )

    meas_df = add_noise_to_measurements(meas_df, noise_level=noise_level)

    PetabIO.write_meas_df(meas_df, filename=paths.measurements())

    return SimulatedExperiment.load(paths, model)


def simulate_experiment(
    model: Type[ModelInterface],
    true_params: ParameterSet,
    conditions: SimulationConditions,
    obs_df: Optional[pd.DataFrame] = None,
    base_dir: str = "data",
) -> SimulatedExperiment:
    """
    Simulate an experiment using a model and parameter set.

    Parameters
    ----------
    model : Type[ModelInterface]
        Model class to use for simulation
    true_params : ParameterSet
        True parameter values for simulation
    conditions : SimulationConditions
        Conditions for the simulation, including time points and
        experimental conditions
    base_dir : str, optional
        Base directory for data, by default "data"

    Returns
    -------
    Experiment
        Experiment object with simulated data
    """
    # Create PetabData from configuration
    if obs_df is None:
        obs_df = model.get_default_observables()
    cond_df = model.create_conditions(**conditions.conditions)
    meas_df = define_empty_measurements(obs_df, cond_df, conditions.t_eval)

    if conditions.fit_params is None:
        conditions.fit_params = model.get_default_parameters()

    petab_data = PetabData(
        obs_df=obs_df,
        cond_df=cond_df,
        param_df=conditions.fit_params,
        meas_df=meas_df,
    )

    # Create experiment directory name based on configuration
    exp_dir = os.path.join(base_dir, conditions.name)

    # Create problem set with simulated data
    paths = ExperimentPaths(exp_dir, true_params.id)

    return create_simulated_experiment(
        model=model,
        true_params=true_params,
        data=petab_data,
        paths=paths,
        noise_level=conditions.noise_level,
    )
