import os
from pathlib import Path
from typing import Dict, List


class ExperimentPaths:
    """
    Manages pathing for experiment data.
    - Petab Data
    - PyPESTO Results
    - Figures etc.
    """

    def __init__(self, base_dir: str | Path, exp_id: str):
        self.base_dir = Path(base_dir)
        self.exp_id = exp_id
        self.make_dirs()

    def make_dirs(self) -> None:
        """Create all necessary directories."""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.petab_dir, exist_ok=True)
        os.makedirs(self.common_dir, exist_ok=True)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.pypesto_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    ##########################
    ### Common Petab Paths ###
    ##########################

    @property
    def petab_dir(self) -> str:
        return f"{self.base_dir}/petab"

    @property
    def common_dir(self) -> str:
        return f"{self.petab_dir}/common"

    @property
    def conditions(self) -> str:
        return f"{self.common_dir}/conditions.tsv"

    @property
    def observables(self) -> str:
        return f"{self.common_dir}/observables.tsv"

    @property
    def fit_parameters(self) -> str:
        return f"{self.common_dir}/parameters.tsv"

    def model(self, name: str = "") -> str:
        return f"{self.common_dir}/{name}.xml"

    ######################################
    ### Parameter specific Petab paths ###
    ######################################

    @property
    def exp_dir(self) -> str:
        return f"{self.petab_dir}/{self.exp_id}"

    def true_params(self) -> str:
        return f"{self.exp_dir}/params.json"

    def measurements(self) -> str:
        return f"{self.exp_dir}/measurements.tsv"

    def petab_yaml(self) -> str:
        return f"{self.exp_dir}/petab.yaml"

    ############################
    ### PyPESTO Result Paths ###
    ############################

    @property
    def pypesto_dir(self) -> str:
        return f"{self.base_dir}/pypesto"

    @property
    def results_dir(self) -> str:
        return f"{self.pypesto_dir}/{self.exp_id}"

    @property
    def pypesto_results(self) -> str:
        return f"{self.results_dir}/results.hdf5"

    ####################
    ### Figure Paths ###
    ####################

    @property
    def figures_base_dir(self) -> str:
        return f"{self.base_dir}/figures"

    @property
    def figures_dir(self) -> str:
        return f"{self.figures_base_dir}/{self.exp_id}"

    @property
    def measurements_data_plot(self) -> str:
        return f"{self.figures_dir}/measurements.png"

    @property
    def waterfall_plot(self) -> str:
        return f"{self.figures_dir}/waterfall.png"

    @property
    def profile_plot(self) -> str:
        return f"{self.figures_dir}/profile.png"

    @property
    def sampling_trace_plot(self) -> str:
        return f"{self.figures_dir}/sampling_trace.png"

    @property
    def confidence_intervals_plot(self) -> str:
        return f"{self.figures_dir}/confidence_intervals.png"

    @property
    def sampling_scatter_plot(self) -> str:
        return f"{self.figures_dir}/sampling_scatter.png"

    @property
    def optimization_scatter_plot(self) -> str:
        return f"{self.figures_dir}/optimization_scatter.png"

    @property
    def ensemble_predictions_plot(self) -> str:
        return f"{self.figures_dir}/ensemble_predictions.png"

    ########################
    ### Helper functions ###
    ########################

    def assert_parameters_exist(self):
        if not os.path.exists(self.true_params()):
            raise FileNotFoundError(
                f"True parameters file not found at {self.true_params()}"
            )

    def get_base_name(self) -> str:
        return self.base_dir.name

    def get_exp_id(self) -> str:
        return self.exp_id

    @staticmethod
    def get_base_dir_from_yaml(yaml_path: str | Path) -> str:
        """Get the base directory from a petab.yaml file"""
        yaml_path = Path(yaml_path)
        return yaml_path.parent.parent.parent

    @staticmethod
    def get_exp_id_from_yaml(yaml_path: str | Path) -> str:
        """Get the experiment ID from a petab.yaml file"""
        yaml_path = Path(yaml_path)
        return yaml_path.parent.name

    @staticmethod
    def from_yaml(results_path: str | Path) -> "ExperimentPaths":
        """Create ExperimentPaths from a results.hdf5 file"""
        results_path = Path(results_path)
        return ExperimentPaths(
            base_dir=ExperimentPaths.get_base_dir_from_yaml(results_path),
            exp_id=ExperimentPaths.get_exp_id_from_yaml(results_path),
        )


#             )
