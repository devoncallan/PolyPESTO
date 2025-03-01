import os
from pathlib import Path
from typing import Dict, List

from polypesto.core.params import ParameterSetID


class ExperimentPaths:
    """
    Manages pathing for experiment data.
    - Petab Data
    - PyPESTO Results
    - Figures etc.
    """

    def __init__(self, base_dir: str | Path, p_id: ParameterSetID):
        self.base_dir = Path(base_dir)
        self.p_id = p_id
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
        return f"{self.petab_dir}/{self.p_id}"

    def make_exp_dir(self) -> None:
        os.makedirs(self.exp_dir, exist_ok=True)

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
        return f"{self.pypesto_dir}/{self.p_id}"

    def make_results_dir(self) -> None:
        os.makedirs(self.results_dir, exist_ok=True)

    @property
    def pypesto_results(self) -> str:
        return f"{self.results_dir}/results.hdf5"

    ####################
    ### Figure Paths ###
    ####################

    @property
    def figures_dir(self) -> str:
        return f"{self.base_dir}/figures"

    ########################
    ### Helper functions ###
    ########################

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "ExperimentPaths":
        """Create ExperimentPaths from a petab.yaml file"""
        yaml_path = Path(yaml_path)
        return ExperimentPaths(
            base_dir=yaml_path.parent.parent.parent, p_id=yaml_path.parent.name
        )


def find_yaml_paths(base_dir: str) -> Dict[ParameterSetID, List[str]]:
    """Find all petab.yaml files and map them to their parameter set IDs"""
    base_dir = Path(base_dir)

    yaml_paths = {}
    for yaml_path in base_dir.glob("**/petab.yaml"):
        p_id = yaml_path.parent.name
        
        if p_id not in yaml_paths:
            yaml_paths[p_id] = []
        yaml_paths[p_id].append(str(yaml_path))

    # yaml_paths = dict(sorted(yaml_paths.keys()))
    return yaml_paths
