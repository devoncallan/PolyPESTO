"""
Utilities for loading and managing experiment data.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import pypesto
from pypesto.petab import PetabImporter
from pypesto.problem import Problem

from polypesto.utils._paths import PetabPaths
from polypesto.utils.file import read_json
from polypesto.core.params import ParameterGroup, ParameterSet


@dataclass
class ExperimentData:
    """
    Container for experiment data loaded from a directory.

    Attributes
    ----------
    parameter_group : ParameterGroup
        Group of true parameters used for this experiment
    petab_paths : PetabPaths
        Object containing paths to all experiment files
    problems : Dict[str, Tuple[PetabImporter, Problem]]
        Dictionary mapping parameter set IDs to (importer, problem) tuples
    results : Dict[str, pypesto.Result], optional
        Dictionary mapping parameter set IDs to optimization results
    """

    parameter_group: ParameterGroup
    petab_paths: PetabPaths
    problems: Dict[str, Tuple[PetabImporter, Problem]]
    results: Dict[str, pypesto.Result] = None

    @property
    def param_ids(self) -> List[str]:
        """Get list of all parameter set IDs"""
        return list(self.problems.keys())

    def get_problem(self, param_id: str) -> Tuple[PetabImporter, Problem]:
        """Get problem for a specific parameter set"""
        if param_id not in self.problems:
            raise KeyError(f"Parameter set {param_id} not found")
        return self.problems[param_id]

    def get_true_params(self, param_id: str) -> ParameterSet:
        """Get true parameters for a specific parameter set"""
        return self.parameter_group.by_id(param_id)

    def get_result(self, param_id: str) -> pypesto.Result:
        """Get optimization result for a specific parameter set.

        If the result is not already loaded in memory, tries to load it from disk.

        Parameters
        ----------
        param_id : str
            ID of the parameter set

        Returns
        -------
        pypesto.Result
            The optimization result

        Raises
        ------
        ValueError
            If no result is available for the specified parameter set
        """
        if self.results is None or param_id not in self.results:
            # Try to load the result
            try:
                result_path = self.petab_paths.pypesto_results(param_id)
                if os.path.exists(result_path):
                    if self.results is None:
                        self.results = {}
                    self.results[param_id] = pypesto.store.read_result(result_path)
            except Exception as e:
                raise ValueError(
                    f"Could not load result for parameter set {param_id}: {e}"
                )

        if self.results is not None and param_id in self.results:
            return self.results[param_id]
        else:
            raise ValueError(f"No result available for parameter set {param_id}")

    def load_all_results(self) -> None:
        """Load all available results from disk.

        This method attempts to load results for all parameter sets.
        It silently skips parameter sets that don't have results.
        """
        if self.results is None:
            self.results = {}

        for param_id in self.param_ids:
            try:
                result_path = self.petab_paths.pypesto_results(param_id)
                if os.path.exists(result_path):
                    self.results[param_id] = pypesto.store.read_result(result_path)
            except Exception as e:
                print(
                    f"Warning: Could not load result for parameter set {param_id}: {e}"
                )

    def has_result(self, param_id: str) -> bool:
        """Check if a result is available for a parameter set.

        Parameters
        ----------
        param_id : str
            ID of the parameter set

        Returns
        -------
        bool
            True if a result is available, False otherwise
        """
        if self.results is not None and param_id in self.results:
            return True

        # Check if result exists on disk
        result_path = self.petab_paths.pypesto_results(param_id)
        return os.path.exists(result_path)

    def get_best_parameters(self, param_id: str) -> Dict[str, float]:
        """Get the best fit parameters for a parameter set.

        Parameters
        ----------
        param_id : str
            ID of the parameter set

        Returns
        -------
        Dict[str, float]
            Dictionary mapping parameter names to values

        Raises
        ------
        ValueError
            If no result is available for the specified parameter set
        """
        result = self.get_result(param_id)

        # Get parameter names
        importer, problem = self.get_problem(param_id)
        parameter_names = problem.x_names

        # Get best parameters
        best_params = result.optimize_result.get_for_best()
        best_x = best_params.x

        # Create dictionary
        return dict(zip(parameter_names, best_x))

    def visualize_optimization_waterfall(self, param_id: str, ax=None):
        """Visualize the optimization waterfall plot for a parameter set.

        Parameters
        ----------
        param_id : str
            ID of the parameter set
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the waterfall plot
        """
        import pypesto.visualize as visualize

        result = self.get_result(param_id)
        return visualize.waterfall(result, ax=ax)

    def visualize_parameter_profiles(self, param_id: str, ax=None):
        """Visualize the parameter profiles for a parameter set.

        Parameters
        ----------
        param_id : str
            ID of the parameter set
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the parameter profiles
        """
        import pypesto.visualize as visualize

        result = self.get_result(param_id)
        if result.profile_result is None:
            raise ValueError(
                f"No profile results available for parameter set {param_id}"
            )

        return visualize.profiles(result, ax=ax)

    def visualize_sampling_traces(self, param_id: str, ax=None):
        """Visualize the sampling traces for a parameter set.

        Parameters
        ----------
        param_id : str
            ID of the parameter set
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the sampling traces
        """
        import pypesto.visualize as visualize

        result = self.get_result(param_id)
        if result.sample_result is None:
            raise ValueError(
                f"No sampling results available for parameter set {param_id}"
            )

        return visualize.sampling_parameter_traces(result, ax=ax)


def load_experiment(exp_dir: str, model_name: str) -> ExperimentData:
    """
    Load experiment data including true parameter group and all problems.

    Parameters
    ----------
    exp_dir : str
        Path to experiment directory (e.g., ".../data/fA0_0.10/")
    model_name : str
        Name of the model to use when loading problems

    Returns
    -------
    ExperimentData
        Object containing all experiment data
    """
    # Create paths object
    petab_paths = PetabPaths(exp_dir)

    # Load parameter group
    try:
        parameter_group = ParameterGroup.load(petab_paths.true_params)
    except (FileNotFoundError, ValueError) as e:
        # Fall back to reconstructing from individual param files
        print(
            f"Warning: Could not load parameter group from {petab_paths.true_params}: {e}"
        )
        parameter_group = _reconstruct_parameter_group(petab_paths)

    # Find all YAML paths
    yaml_paths = petab_paths.find_yaml_paths()

    # Load all problems
    from polypesto.core.pypesto import load_pypesto_problem

    problems = {}
    for param_id, yaml_path in yaml_paths.items():
        importer, problem = load_pypesto_problem(
            yaml_path=yaml_path, model_name=model_name
        )
        problems[param_id] = (importer, problem)

    # Load results for all parameter sets
    results = {}
    for param_id in yaml_paths.keys():
        try:
            result_path = petab_paths.pypesto_results(param_id)
            if os.path.exists(result_path):
                results[param_id] = pypesto.store.read_result(result_path)
        except Exception as e:
            print(f"Warning: Could not load result for parameter set {param_id}: {e}")

    return ExperimentData(
        parameter_group=parameter_group,
        petab_paths=petab_paths,
        problems=problems,
        results=results if results else None,
    )


def load_all_experiments(data_dir: str, model_name: str) -> Dict[str, ExperimentData]:
    """
    Load all experiments in the given directory.

    Parameters
    ----------
    data_dir : str
        Base directory containing all experiment directories
    model_name : str
        Name of the model to use when loading problems

    Returns
    -------
    Dict[str, ExperimentData]
        Dictionary of experiment directory name to experiment data
    """
    experiments = {}

    # Find all directories in the data directory
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Check if this is an experiment directory by looking for petab dir
            petab_dir = os.path.join(item_path, "petab")
            if os.path.exists(petab_dir):
                try:
                    # Load experiment data
                    experiments[item] = load_experiment(item_path, model_name)
                    print(f"Loaded experiment: {item}")
                except Exception as e:
                    print(f"Error loading experiment {item}: {e}")

    return experiments


def _reconstruct_parameter_group(petab_paths: PetabPaths) -> ParameterGroup:
    """
    Reconstruct a parameter group from individual parameter set files.

    This is a fallback method if the true_params file doesn't exist.

    Parameters
    ----------
    petab_paths : PetabPaths
        Paths object for the experiment

    Returns
    -------
    ParameterGroup
        Reconstructed parameter group
    """
    yaml_paths = petab_paths.find_yaml_paths()
    parameter_sets = {}

    # For each parameter set, load the params.json file
    for param_id in yaml_paths.keys():
        params_file = petab_paths.params(param_id)
        if os.path.exists(params_file):
            parameter_set = ParameterSet.load(params_file)
            parameter_sets[param_id] = parameter_set

    # Create a new parameter group
    pg = ParameterGroup(id="reconstructed", parameter_sets=parameter_sets)
    return pg
