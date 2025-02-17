from dataclasses import dataclass
import os
import re
from typing import Dict, Any, List, Sequence, Optional
import pandas as pd

import src.utils.petab as pet
from src.utils.params import ParameterSetID, ParameterGroup
from src.utils.file import Filename, Filepath, Directory


PETAB_DIRNAME = "petab"
PLOTS_DIRNAME = "plots"

MEAS_BASE_FILENAME = "meas_{}.tsv"
COND_FILENAME = "cond.tsv"
OBS_FILENAME = "obs.tsv"
TRUE_PARAMS_FILENAME = "params.json"


# DatasetPaths should define the database structure for a petab dataset
# This is what determines how the dataset is stored on disk
# And what constitutes a valid dataset

# SimulatedDataset
# SimulatedDatasetPaths


@dataclass
class DatasetPaths:

    ds_dir: Directory
    param_set_ids: List[ParameterSetID]
    name: Optional[str] = None
    model_filepath: Optional[str] = None

    def __post_init__(self):

        #### PETAB ####
        self.petab_dir = os.path.join(self.ds_dir, PETAB_DIRNAME)

        self.obs_filepath = os.path.join(self.petab_dir, OBS_FILENAME)
        self.cond_filepath = os.path.join(self.petab_dir, COND_FILENAME)
        self.true_params_filepath = os.path.join(self.petab_dir, TRUE_PARAMS_FILENAME)

        self.meas_filepaths_dict = {
            id: os.path.join(self.petab_dir, MEAS_BASE_FILENAME.format(id))
            for id in self.param_set_ids
        }

        #### PLOTS ####
        self.plots_dir = os.path.join(self.ds_dir, PLOTS_DIRNAME)

    def make_dirs(self):
        os.makedirs(self.ds_dir, exist_ok=True)
        os.makedirs(self.petab_dir, exist_ok=True)
        # os.makedirs(self.plots_dir, exist_ok=True)

    def validate(self):
        assert os.path.exists(self.obs_filepath)
        assert os.path.exists(self.cond_filepath)
        assert os.path.exists(self.true_params_filepath)
        # assert os.path.exists(self.model_path)
        for id, meas_filepath in self.meas_filepaths_dict.items():
            assert os.path.exists(meas_filepath)

    @staticmethod
    def from_dir(ds_dir: str) -> "DatasetPaths":

        # Search for the model file in the dataset directory
        model_filepath = None
        for file in os.listdir(ds_dir):
            if file.endswith(".xml"):
                model_filepath = os.path.join(ds_dir, file)
                break

        # Search for the parameter set ids from measurement dataframes in the petab directory
        petab_dir = os.path.join(ds_dir, PETAB_DIRNAME)
        filepaths = [os.path.join(petab_dir, f) for f in os.listdir(petab_dir)]

        meas_re_pattern = re.escape(MEAS_BASE_FILENAME).replace(
            r"\{\}", r"(.*?)"
        )  # Replace `{}` with a regex group
        meas_re_pattern = rf".*/{meas_re_pattern}"  # Match full paths

        # Compile the regex
        meas_re = re.compile(meas_re_pattern)

        # Extract the dynamic parts
        param_set_ids = [
            match.group(1)
            for filepath in filepaths
            if (match := meas_re.match(filepath))
        ]

        ds_name = os.path.basename(ds_dir)
        paths = DatasetPaths(
            ds_dir=ds_dir,
            param_set_ids=param_set_ids,
            name=ds_name,
            model_filepath=model_filepath,
        )

        paths.validate()

        return paths


@dataclass
class PetabDataset:

    name: str
    obs_df: pd.DataFrame
    cond_df: pd.DataFrame
    param_group: ParameterGroup
    meas_dfs: Dict[ParameterSetID, pd.DataFrame]

    model_filepath: Optional[str] = None
    paths: Optional[DatasetPaths] = None

    def write(self, ds_dir: Directory) -> "PetabDataset":
        self.paths = DatasetPaths(
            ds_dir=ds_dir,
            param_set_ids=list(self.meas_dfs.keys()),
            name=self.name,
            model_filepath=self.model_filepath,
        )

        self.paths.make_dirs()

        pet.PetabIO.write_obs_df(self.obs_df, filename=self.paths.obs_filepath)
        pet.PetabIO.write_cond_df(self.cond_df, filename=self.paths.cond_filepath)
        for id, meas_df in self.meas_dfs.items():
            pet.PetabIO.write_meas_df(
                meas_df, filename=self.paths.meas_filepaths_dict[id]
            )

        self.param_group.write(self.paths.true_params_filepath)

        return self

    @staticmethod
    def load(ds_dir: str) -> "PetabDataset":

        paths = DatasetPaths.from_dir(ds_dir)

        obs_df = pet.PetabIO.read_obs_df(paths.obs_filepath)
        cond_df = pet.PetabIO.read_cond_df(paths.cond_filepath)
        param_group = ParameterGroup.load(paths.true_params_filepath)
        meas_dfs = {
            id: pet.PetabIO.read_meas_df(meas_fp)
            for id, meas_fp in paths.meas_filepaths_dict.items()
        }

        return PetabDataset(
            name=paths.name,
            obs_df=obs_df,
            cond_df=cond_df,
            param_group=param_group,
            meas_dfs=meas_dfs,
            paths=paths,
            model_filepath=paths.model_filepath,
        )
