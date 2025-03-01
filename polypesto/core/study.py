from typing import Type, TypeAlias, Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import os
from pathlib import Path
import json
import shutil

import pandas as pd
from pypesto import Result
from pypesto.store import read_result

from polypesto.models import ModelInterface
from polypesto.core.params import ParameterGroup, ParameterSetID, ParameterSet
from polypesto.core.experiment import Experiment, ExperimentConfig, simulate_experiment
from polypesto.utils.paths import ExperimentPaths, find_yaml_paths
from polypesto.core.pypesto import run_parameter_estimation, PEConfigDict


TrialID: TypeAlias = str
ResultDict: TypeAlias = Dict[ParameterSetID, Result]


@dataclass
class StudyExperiment:
    """
    Container for an experiment within a study.

    Parameters
    ----------
    trial_id : TrialID
        Identifier for the trial (experiment configuration)
    p_id : ParameterSetID
        Identifier for the parameter set
    experiment : Experiment
        The experiment object
    """

    trial_id: TrialID
    p_id: ParameterSetID
    experiment: Experiment


@dataclass
class Study:
    """
    Collection of related experiments for
    understanding parameter estimation.

    Parameters
    ----------
    num_trials : int
        Number of different experimental conditions
    trial_names : List[str]
        Names of the trials/configurations
    model : Type[ModelInterface]
        Model class used for the study
    true_params : ParameterGroup
        Group of true parameter sets used in the study
    experiments : List[StudyExperiment]
        List of experiments in the study
    base_dir : str, optional
        Base directory where the study data is stored
    results : Dict[TrialID, ResultDict], optional
        Dictionary of parameter estimation results indexed by trial ID and parameter set ID
    """

    num_trials: int
    trial_names: List[str]
    model: Type[ModelInterface]
    true_params: ParameterGroup
    experiments: List[StudyExperiment]
    base_dir: Optional[str] = None
    results: Optional[Dict[TrialID, ResultDict]] = None

    def run_parameter_estimation(self, config: Optional[PEConfigDict] = None) -> Dict:
        """
        Run parameter estimation for all experiments in the study.

        This method runs parameter estimation for each experiment in the study
        using the specified configuration. Results are stored in the study object
        and can be saved to disk.

        Parameters
        ----------
        config : Optional[PEConfigDict], optional
            Configuration for parameter estimation steps. Can contain keys
            'optimize', 'profile', 'sample' with respective options.
            Example: {'optimize': {'n_starts': 50, 'method': 'scipy-lbfgsb'}}

        Returns
        -------
        Dict
            Dictionary of parameter estimation results indexed by trial ID and parameter set ID
        """
        # Initialize results storage if needed
        if self.results is None:
            self.results = {}

        # Create experiment collection for parameter estimation
        experiment_collection = {}
        for exp in self.experiments:
            if exp.trial_id not in experiment_collection:
                experiment_collection[exp.trial_id] = {}
            experiment_collection[exp.trial_id][exp.p_id] = exp.experiment

        # Run parameter estimation using the existing function
        results = run_parameter_estimation(experiment_collection, config)

        # Store results in study
        for trial_id, trial_results in results.items():
            if trial_id not in self.results:
                self.results[trial_id] = {}
            self.results[trial_id].update(trial_results)

        return results

    def save(self, dir_path: Optional[str] = None) -> str:
        """
        Save the study to a directory.

        Parameters
        ----------
        dir_path : str, optional
            Directory to save the study to. If not provided,
            uses the study's base_dir or raises an error.

        Returns
        -------
        str
            Path to the saved study directory
        """
        # Determine save directory
        save_dir = dir_path or self.base_dir
        if save_dir is None:
            raise ValueError("No directory specified for saving study")

        os.makedirs(save_dir, exist_ok=True)

        # Save study metadata
        metadata = {
            "num_trials": self.num_trials,
            "trial_names": self.trial_names,
            "model_name": self.model.name,
            "parameter_sets": self.true_params.get_ids(),
        }

        # Write metadata file
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # # Add results metadata if available
        # if not self.results:
        #     return save_dir
        
        # result_summary = {}
        # for trial_id, trial_results in self.results.items():
        #     result_summary[trial_id] = {}
        #     for p_id, result in trial_results.items():
        #         # Get result metadata safely
        #         best_ll = None
        #         if hasattr(result, "optimize_result") and result.optimize_result:
        #             if (
        #                 hasattr(result.optimize_result, "fval")
        #                 and len(result.optimize_result.fval) > 0
        #             ):
        #                 best_ll = float(result.optimize_result.fval[0])

        #         result_summary[trial_id][p_id] = {
        #             "best_ll": best_ll,
        #             "has_profile": hasattr(result, "profile_result")
        #             and result.profile_result is not None,
        #             "has_samples": hasattr(result, "sample_result")
        #             and result.sample_result is not None,
        #         }

        # with open(os.path.join(save_dir, "result_summary.json"), "w") as f:
        #     json.dump(result_summary, f, indent=2)

        # Note: The actual experiment data is already saved in the directory structure
        # when it was created, so we don't need to save it again.

        return save_dir

    def get_experiment(self, trial_id: str, p_id: str) -> Optional[Experiment]:
        """
        Get a specific experiment by trial ID and parameter set ID.

        Parameters
        ----------
        trial_id : str
            The trial ID to look for
        p_id : str
            The parameter set ID to look for

        Returns
        -------
        Optional[Experiment]
            The experiment if found, None otherwise
        """
        for study_exp in self.experiments:
            if study_exp.trial_id == trial_id and study_exp.p_id == p_id:
                return study_exp.experiment
        return None

    def get_experiments_by_trial(self, trial_id: str) -> List[StudyExperiment]:
        """
        Get all experiments for a specific trial.

        Parameters
        ----------
        trial_id : str
            The trial ID to look for

        Returns
        -------
        List[StudyExperiment]
            List of experiments for the trial
        """
        return [exp for exp in self.experiments if exp.trial_id == trial_id]

    def get_experiments_by_parameter(self, p_id: str) -> List[StudyExperiment]:
        """
        Get all experiments for a specific parameter set.

        Parameters
        ----------
        p_id : str
            The parameter set ID to look for

        Returns
        -------
        List[StudyExperiment]
            List of experiments for the parameter set
        """
        return [exp for exp in self.experiments if exp.p_id == p_id]

    def get_result(self, trial_id: str, p_id: str) -> Optional[Result]:
        """
        Get a parameter estimation result for a specific experiment.

        Parameters
        ----------
        trial_id : str
            The trial ID to look for
        p_id : str
            The parameter set ID to look for

        Returns
        -------
        Optional[Result]
            The parameter estimation result if found, None otherwise
        """
        if self.results is None:
            return None
        if trial_id not in self.results:
            return None
        if p_id not in self.results[trial_id]:
            return None
        return self.results[trial_id][p_id]

    # def compare_parameters(self, param_id: str = None) -> pd.DataFrame:
    #     """
    #     Compare parameter estimation results across all experiments.

    #     Uses the analysis module's comparison function to create a
    #     comprehensive DataFrame with parameters across all experiments.

    #     Parameters
    #     ----------
    #     param_id : str, optional
    #         Parameter set ID to use for comparison. If not provided,
    #         uses the first parameter set in the study.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         DataFrame with conditions, true parameters, estimated parameters, etc.
    #     """
    #     from polypesto.analysis.comparison import create_parameter_comparison_df

    #     if self.base_dir is None:
    #         raise ValueError("Study must have a base directory to compare parameters")

    #     # If param_id not provided, use the first parameter set
    #     if param_id is None:
    #         param_id = self.true_params.get_ids()[0]

    #     # Create and return the comparison DataFrame
    #     return create_parameter_comparison_df(
    #         data_dir=self.base_dir, model_name=self.model.name, param_id=param_id
    #     )

    # def visualize_comparison(self, param_id: str = None, **kwargs):
    #     """
    #     Create visualizations comparing parameter estimation results across experiments.

    #     Parameters
    #     ----------
    #     param_id : str, optional
    #         Parameter set ID to use for comparison. If not provided,
    #         uses the first parameter set in the study.
    #     **kwargs : dict
    #         Additional visualization options to pass to the comparison function.

    #     Returns
    #     -------
    #     tuple
    #         (DataFrame with comparison data, dict with visualization figures)
    #     """
    #     from polypesto.analysis.comparison import compare_parameters_across_conditions

    #     if self.base_dir is None:
    #         raise ValueError("Study must have a base directory to compare parameters")

    #     # If param_id not provided, use the first parameter set
    #     if param_id is None:
    #         param_id = self.true_params.get_ids()[0]

    #     # Create comparison visualizations
    #     return compare_parameters_across_conditions(
    #         data_dir=self.base_dir,
    #         model_name=self.model.name,
    #         param_id=param_id,
    #         **kwargs,
    #     )

    @staticmethod
    def load(dir_path: str, model: Type[ModelInterface]) -> "Study":
        """Load a study from a directory."""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Study directory not found: {dir_path}")

        # Load metadata
        metadata = {}
        metadata_path = os.path.join(dir_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        # Load experiments
        experiments: List[StudyExperiment] = []
        parameter_sets: Dict[ParameterSetID, ParameterSet] = {}
        yaml_paths = find_yaml_paths(dir_path)
        print(yaml_paths)

        for p_id, yaml_paths in yaml_paths.items():
            for yaml_path in yaml_paths:
                paths = ExperimentPaths.from_yaml(yaml_path)
                experiment = Experiment.load(paths, model)
                exp_name = paths.base_dir.name

                experiments.append(StudyExperiment(exp_name, p_id, experiment))
                parameter_sets[p_id] = experiment.true_params

        # Create parameter group
        study_name = os.path.basename(dir_path)
        param_group = ParameterGroup(id=study_name, parameter_sets=parameter_sets)

        # Get trial names
        trial_names = metadata.get(
            "trial_names", sorted(set(exp.trial_id for exp in experiments))
        )
        num_trials = metadata.get("num_trials", len(set(trial_names)))

        # Load results directly by checking for files
        results = {}
        for exp in experiments:
            trial_id = exp.trial_id
            p_id = exp.p_id

            if trial_id not in results:
                results[trial_id] = {}

            result_path = exp.experiment.paths.pypesto_results
            if os.path.exists(result_path):
                try:
                    results[trial_id][p_id] = read_result(result_path)
                except Exception as e:
                    print(f"Failed to load result for {trial_id}/{p_id}: {e}")

        return Study(
            num_trials=num_trials,
            trial_names=trial_names,
            model=model,
            true_params=param_group,
            experiments=experiments,
            base_dir=dir_path,
            results=results,
        )


def create_study(
    model: Type[ModelInterface],
    true_params: ParameterGroup,
    configs: List[ExperimentConfig],
    base_dir: str = "data",
    overwrite: bool = False,
) -> Study:
    """
    Create a study object with multiple experiments.

    Parameters
    ----------
    model : Type[ModelInterface]
        Model class to use for simulations
    true_params : ParameterGroup
        Group of parameter sets to use as true values
    configs : List[ExperimentConfig]
        List of experiment configurations
    base_dir : str, optional
        Base directory for storing data, by default "data"
    overwrite : bool, optional
        Whether to overwrite existing data, by default False

    Returns
    -------
    Study
        Study object with all experiments
    """
    num_trials = len(configs)
    os.makedirs(base_dir, exist_ok=True)

    if overwrite:
        shutil.rmtree(base_dir, ignore_errors=True)
        os.makedirs(base_dir, exist_ok=True)

    experiments = []
    for i, config in enumerate(configs):
        for p_id in true_params.get_ids():

            experiment = simulate_experiment(
                model=model,
                true_params=true_params.by_id(p_id),
                config=config,
                base_dir=base_dir,
            )

            experiments.append(
                StudyExperiment(trial_id=config.name, p_id=p_id, experiment=experiment)
            )

    return Study(
        num_trials=num_trials,
        trial_names=[config.name for config in configs],
        model=model,
        true_params=true_params,
        experiments=experiments,
        base_dir=base_dir,
    )
