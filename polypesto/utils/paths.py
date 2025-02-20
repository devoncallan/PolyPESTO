import os
from pathlib import Path
from typing import Dict

class PetabPaths:
    """Manages pathing for PEtab data"""
    
    def __init__(self, base_dir: str | Path):
        self.base_dir = str(Path(base_dir))
        self.make_dirs()
        
    def make_dirs(self):
        # Should be data directory
        parent_dir = os.path.dirname(self.base_dir)
        # Should be model directory
        grandparent_dir = os.path.dirname(parent_dir)
        
        os.makedirs(grandparent_dir, exist_ok=True)
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.petab_dir, exist_ok=True)
        os.makedirs(self.common_dir, exist_ok=True)

    ### Directories

    @property
    def petab_dir(self) -> str:
        return f"{self.base_dir}/petab"

    @property
    def common_dir(self) -> str:
        return f"{self.petab_dir}/common"

    def exp_dir(self, p_id: str) -> str:
        return f"{self.petab_dir}/{p_id}"
    
    def make_exp_dir(self, p_id: str):
        os.makedirs(self.exp_dir(p_id), exist_ok=True)

    ### Common paths

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

    ### Experiment specific paths

    def params(self, p_id: str) -> str:
        return f"{self.exp_dir(p_id)}/params.json"

    def petab_yaml(self, p_id: str) -> str:
        return f"{self.exp_dir(p_id)}/petab.yaml"

    def measurements(self, p_id: str) -> str:
        return f"{self.exp_dir(p_id)}/measurements.tsv"

    ### Helper functions

    def find_yaml_paths(self) -> Dict[str, str]:
        """Find all petab.yaml files and map them to their parameter IDs"""
        base_path = Path(self.petab_dir)
        yaml_paths = {}
        for yaml_path in base_path.glob("**/petab.yaml"):
            p_id = yaml_path.parent.name
            yaml_paths[p_id] = str(yaml_path)
        return yaml_paths


class PyPestoPaths:
    """Manages pathing for PyPESTO data"""

    def __init__(self, name: str, base_path: str | Path):
        self.name = name
        self.base_dir = Path(base_path)

    @property
    def pypesto_dir(self) -> str:
        return f"{self.base_dir}/pypesto"
    
    @property
    def figures_dir(self) -> str:
        return f"{self.base_dir}/figures"
