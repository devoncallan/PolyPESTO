from typing import Optional, Type, Dict, List
from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd
from pypesto.petab import PetabImporter
from pypesto.problem import Problem as PypestoProblem
from petab.v1 import Problem as PetabProblem
from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df

from polypesto.core.params import ParameterSet
from polypesto.core.pypesto import load_pypesto_problem
from polypesto.models import ModelInterface, sbml
from polypesto.core.petab import PetabData, define_empty_measurements, PetabIO
from polypesto.utils.paths import ExperimentPaths


@dataclass
class Experiment:
    """
    Represents experimental data for a
    single parameter estimation problem.
    """

    petab_problem: PetabProblem
    pypesto_problem: PypestoProblem
    true_params: Optional[ParameterSet] = None
    paths: Optional[ExperimentPaths] = None

    @staticmethod
    def load(paths: ExperimentPaths, model: ModelInterface) -> "Experiment":
        """
        Load an experiment.

        Parameters
        ----------
        paths : ExperimentPaths
            Paths object containing locations of experiment files
        model : ModelInterface
            Model class to use for simulation

        Returns
        -------
        Experiment
            Loaded experiment object
        """
        importer, problem = load_pypesto_problem(
            yaml_path=paths.petab_yaml(), model_name=model.name
        )
        true_params = ParameterSet.load(paths.true_params())

        return Experiment(
            petab_problem=importer.petab_problem,
            pypesto_problem=problem,
            true_params=true_params,
            paths=paths,
        )


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment.

    Parameters
    ----------
    name : str
        Name of the experiment
    t_eval : List[float]
        Time points for evaluation
    conditions : Dict[str, List[float]]
        Experimental conditions (e.g., feed fractions, concentrations)
    fit_params : pd.DataFrame
        Parameter dataframe for fitting
    noise_level : float
        Noise level for measurements
    """

    name: str
    t_eval: List[float]
    conditions: Dict[str, List[float]]
    fit_params: pd.DataFrame
    noise_level: float


def create_experiment_configs(config_dict: Dict) -> List[ExperimentConfig]:
    """
    Convert a dictionary of lists into a list of ExperimentConfig objects.

    Parameters
    ----------
    config_dict : Dict
        Dictionary where each key is a parameter name and each value is a list
        of parameter values for each trial

    Returns
    -------
    List[ExperimentConfig]
        List of ExperimentConfig objects
    """
    # Check that all lists have the same length
    ntrials = len(next(iter(config_dict.values())))
    for key, value in config_dict.items():
        if len(value) != ntrials:
            raise ValueError(
                f"Length mismatch for {key}: expected {ntrials}, got {len(value)}"
            )

    # Create ExperimentConfig objects
    configs = []
    for i in range(ntrials):

        config_kwargs = {}
        for key, value in config_dict.items():
            if key == "conditions" and isinstance(value, dict):
                config_kwargs[key] = {k: v[i] for k, v in value.items()}
            else:
                config_kwargs[key] = value[i]

        config = ExperimentConfig(**config_kwargs)
        configs.append(config)

    return configs


def create_single_problem_set(
    model: Type[ModelInterface],
    true_params: ParameterSet,
    data: PetabData,
    base_dir: str,
    force_compile: bool = False,
) -> ExperimentPaths:
    """Create Petab problem set for a single parameter set.

    Parameters
    ----------
    model : Type[ModelInterface]
        Model interface to use for simulation
    true_params : ParameterSet
        Parameter set to generate data
    data : PetabData
        Contains observables, conditions, measurements, fit params
    base_dir : str
        Directory to write files to
    force_compile : bool, optional
        Force recompilation of model, by default False

    Returns
    -------
    ExperimentPaths
        Object containing paths to all created files
    """
    # Create paths object
    paths = ExperimentPaths(base_dir, true_params.id)
    paths.make_dirs()

    # Write SBML model
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

    # Load problem for simulation
    importer, problem = load_pypesto_problem(
        yaml_path=paths.petab_yaml(), model_name=model.name, force_compile=force_compile
    )

    sim_data = simulate_petab(
        petab_problem=importer.petab_problem,
        amici_model=problem.objective.amici_model,
        solver=problem.objective.amici_solver,
        problem_parameters=true_params.to_dict(),
    )

    meas_df = rdatas_to_measurement_df(
        sim_data["rdatas"],
        problem.objective.amici_model,
        importer.petab_problem.measurement_df,
    )

    # Write measurement data
    PetabIO.write_meas_df(meas_df, filename=paths.measurements())

    return paths


def simulate_experiment(
    model: Type[ModelInterface],
    true_params: ParameterSet,
    config: ExperimentConfig,
    base_dir: str = "data",
) -> Experiment:
    """
    Simulate an experiment using a model and parameter set.

    Parameters
    ----------
    model : Type[ModelInterface]
        Model class to use for simulation
    true_params : ParameterSet
        True parameter values for simulation
    config : ExperimentConfig
        Configuration for the experiment
    base_dir : str, optional
        Base directory for data, by default "data"

    Returns
    -------
    Experiment
        Experiment object with simulated data
    """
    # Create PetabData from configuration
    obs_df = model.get_default_observables()
    cond_df = model.create_conditions(**config.conditions)
    meas_df = define_empty_measurements(obs_df, cond_df, config.t_eval)

    petab_data = PetabData(
        obs_df=obs_df,
        cond_df=cond_df,
        param_df=config.fit_params,
        meas_df=meas_df,
    )

    # Create experiment directory name based on configuration
    exp_dir = os.path.join(base_dir, config.name)

    # Create problem set with simulated data
    paths = create_single_problem_set(
        model=model, true_params=true_params, data=petab_data, base_dir=exp_dir
    )

    # Load the created experiment
    return Experiment.load(paths, model)


"""


def run_study(
    cls,
    model,
    true_params,
    fit_params,
    conditions,
    time_points,
    noise_levels,
    trial_names,
):

    num_trials = len(fit_params)
    assert len(conditions.values()[0]) == num_trials
    assert len(time_points) == num_trials
    assert len(noise_levels) == num_trials
    
    for i in range(num_trials):
    
        param_set = true__params[i]
        fit_params = fit_params[i]
        time_points = time_points[i]
        noise_level = noise_levels[i]
        
        for param_set in true_params:
        
            experiment = simulate_experiment(
                model=Model,
                param_set=param_set,
                conditions=conditions,
                time_points=time_points,
                noise_level=0.02
                trial_name=trial_names[i],
                exp_id=...,
                data_dir=...
            )

            
    return cls(
        model=Model,
        true_params=true_params,
        trial_names=trial_names,
        experiments={...},
    )
    
def simulate_experiment(
    model,
    param_set,
    fit_params,
    conditions,
    time_points,
    noise_level,
    trial_name,
    data_dir,
) -> Experiment:

    obs_df = Model.get_default_observables()
    cond_df = Model.create_conditions(conditions)
    param_df = fit_params
    meas_df = pet.define_empty_measurements(obs_df, cond_df, time_points)
    petab_data = pet.PetabData(
        obs_df=Model.get_default_observables(),
        cond_df=Model.create_conditions(condition),
        param_df=fit_params,
        meas_df=empty_meas_df
    )
    
    return create_experiments(Model, param_set, petab_data, data_dir)


Experiment:


    yaml_filepath / petab_problem / petab_data?
    
    
    def load(...):
        ...
        
def load_all_experiments():
    ...
    
"""
