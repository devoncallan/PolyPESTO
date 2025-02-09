from pathlib import Path
from typing import Dict

"""

model:
- exp_0:
    - petab:
        - common
            - conditions.tsv
            - observables.tsv
            - parameters.tsv
            - model.xml
            - all_params.json
        - p_0:
            - params.json
            - measurements.tsv
            - petab.yaml
        - p_1:
            - params.json
            - measurements.tsv
            - petab.yaml
        - ...
    - figures:
        - measurements:
            - p_0_measurements.png
            - p_1_measurements.png
        - pypesto:
            - ...
    - pypesto:
        - ...
- exp_1:
    - ...
"""



class PetabPaths:
    def __init__(self, base_dir: str | Path):
        self.base = str(Path(base_dir))

    ### Directories

    @property
    def petab_dir(self) -> str:
        return f"{self.base}/petab"

    @property
    def common_dir(self) -> str:
        return f"{self.petab_dir}/common"

    def exp_dir(self, p_id: str) -> str:
        return f"{self.petab_dir}/{p_id}"

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

    def __init__(self, name: str, base_path: str | Path):
        self.name = name
        self.base = Path(base_path)

    @property
    def pypesto_dir(self) -> str:
        return f"{self.base}/pypesto"
    
    @property
    def figures_dir(self) -> str:
        return f"{self.base}/figures"
