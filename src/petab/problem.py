from dataclasses import dataclass
from src.utils.file import Directory
import pandas as pd

# from dataset import PetabDataset
from typing import Optional
import os

from typing import Dict, Any, List, Sequence, Optional
import src.utils.petab as pet
from src.utils.file import Filename, Filepath, Directory
from src.petab.dataset import PetabDataset

PETAB_DIRNAME = "petab"
PYPESTO_DIRNAME = "pypesto"
FIGURES_DIRNAME = "figures"

PARAMS_FILENAME = "params.tsv"
PETAB_YAML_BASE = "petab_{}.yaml"


@dataclass
class PetabProblemSetPaths:

    dir: Directory
    dataset: PetabDataset  # Reference to associated dataset
    name: Optional[str] = None

    def __post_init__(self):
        # Main directories
        self.petab_dir = os.path.join(self.dir, PETAB_DIRNAME)
        self.pypesto_dir = os.path.join(self.dir, PYPESTO_DIRNAME)
        self.figures_dir = os.path.join(self.dir, FIGURES_DIRNAME)

        # Core files
        self.fit_params_filepath = os.path.join(self.petab_dir, PARAMS_FILENAME)

        # We'll discover/store paths to all petab yaml files
        self.yaml_paths_dict = {
            id: os.path.join(self.petab_dir, PETAB_YAML_BASE.format(id))
            for id in self.dataset.paths.param_set_ids
        }

    def make_dirs(self):
        """Create all necessary directories"""
        os.makedirs(self.petab_dir, exist_ok=True)
        os.makedirs(self.pypesto_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def validate(self):
        """Check that all necessary files exist"""

        assert os.path.exists(self.fit_params_filepath)
        for id, yaml_path in self.yaml_paths_dict.items():
            assert os.path.exists(yaml_path)

    @staticmethod
    def from_dataset_dir(
        ds_dir: Directory, new_dir: Directory
    ) -> "PetabProblemSetPaths":

        dataset = PetabDataset.load(ds_dir)

        pps_name = os.path.basename(new_dir)
        return PetabProblemSetPaths(dir=new_dir, dataset=dataset, name=pps_name)

    # @staticmethod
    # def from_dir(dir: Directory) -> "PetabProblemSetPaths":
    #     """Reconstruct ProblemSetPaths from existing directory"""
    #     # First we need to find and load the associated dataset
    #     dataset = PetabDataset.load(os.path.join(dir, "dataset"))  # Assuming convention

    #     paths = PetabProblemSetPaths(
    #         dir=dir, dataset=dataset, name=os.path.basename(dir)
    #     )

    #     # Scan for existing yaml configs
    #     yaml_pattern = PETAB_YAML_BASE.replace("{}", r"(\d+)")
    #     for file in os.listdir(paths.petab_dir):
    #         if match := re.match(yaml_pattern, file):
    #             config_id = match.group(1)
    #             paths.yaml_paths[config_id] = os.path.join(paths.petab_dir, file)

    #     return paths


@dataclass
class PetabProblemSet:

    name: str
    param_df: pd.DataFrame  # Parameters to be estimated
    dataset: PetabDataset  # Reference to associated dataset

    paths: Optional[PetabProblemSetPaths] = None

    def write(self, dir: Directory) -> "PetabProblemSet":
        """Write the complete problem set to disk"""
        # Initialize paths if not already set
        if self.paths is None:
            self.paths = PetabProblemSetPaths(
                dir=dir, dataset=self.dataset, name=self.name
            )

        self.paths.make_dirs()
        pet.PetabIO.write_param_df(
            self.param_df, filename=self.paths.fit_params_filepath
        )
        ds_paths = self.dataset.paths

        for id, yaml_path in self.paths.yaml_paths_dict.items():
            pet.PetabIO.write_yaml(
                yaml_filepath=yaml_path,
                sbml_filepath=ds_paths.model_filepath,
                cond_filepath=ds_paths.cond_filepath,
                meas_filepath=ds_paths.meas_filepaths_dict[id],
                obs_filepath=ds_paths.obs_filepath,
                param_filepath=self.paths.fit_params_filepath,
            )
            
        return self

    @staticmethod
    def load(dir: Directory) -> "PetabProblemSet":
        """Load a complete problem set from disk"""

        paths = PetabProblemSetPaths.from_dir(dir)
        param_df = pet.PetabIO.read_param_df(paths.params_path)

        return PetabProblemSet(
            name=paths.name,
            param_df=param_df,
            dataset=paths.dataset,
            paths=paths,
        )
