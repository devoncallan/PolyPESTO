from dataclasses import dataclass
import os
import re
from typing import Dict, Any, List, Sequence, Optional
import pandas as pd

import src.models.amici as am
import src.utils.petab as pet
from src.utils.params import ParameterSetID, ParameterGroup
from src.utils.file import Filename, Filepath, Directory


PETAB_DIRNAME = "petab"
PLOTS_DIRNAME = "plots"

MEAS_BASE_FILENAME = "meas_{}.tsv"
COND_FILENAME = "cond.tsv"
OBS_FILENAME = "obs.tsv"
S_PARAMS_FILENAME = "params.json"
SBML_FILENAME = "model.xml"


# DatasetPaths should define the database structure for a petab dataset
# This is what determines how the dataset is stored on disk
# And what constitutes a valid dataset


@dataclass
class DatasetPaths:

    ds_dir: Directory
    param_set_ids: List[ParameterSetID]
    name: Optional[str] = None

    def __post_init__(self):

        #### DATASET ####
        self.model_path = os.path.join(self.ds_dir, SBML_FILENAME)

        #### PETAB ####
        self.petab_dir = os.path.join(self.ds_dir, PETAB_DIRNAME)

        self.obs_path = os.path.join(self.petab_dir, OBS_FILENAME)
        self.cond_path = os.path.join(self.petab_dir, COND_FILENAME)
        self.tparam_path = os.path.join(self.petab_dir, S_PARAMS_FILENAME)

        self.meas_paths_dict = {
            id: os.path.join(self.petab_dir, MEAS_BASE_FILENAME.format(id))
            for id in self.param_set_ids
        }

        #### PLOTS ####
        self.plots_dir = os.path.join(self.ds_dir, PLOTS_DIRNAME)

    def make_dirs(self):
        os.makedirs(self.ds_dir, exist_ok=True)
        os.makedirs(self.petab_dir, exist_ok=True)
        # os.makedirs(self.plots_dir, exist_ok=True)

    def get_fpath_dict(self, fpath: Filepath = "{}") -> Dict[ParameterSetID, str]:
        return {
            id: os.path.join(self.petab_dir, fpath.format(id))
            for id in self.param_set_ids
        }

    def get_fpaths(self, dir: str, f_filename: str = "{}") -> List[str]:
        return [os.path.join(dir, f_filename.format(id)) for id in self.param_set_ids]

    def search(self, dataset_dir: str):
        # Scan the dataset directory for files
        files = os.listdir(self.dataset)

    def validate(self):
        assert os.path.exists(self.obs_path)
        assert os.path.exists(self.cond_path)
        assert os.path.exists(self.tparam_path)
        # assert os.path.exists(self.model_path)
        for id, meas_fp in self.meas_paths_dict.items():
            assert os.path.exists(meas_fp)

    @staticmethod
    def from_dir(ds_dir: str) -> "DatasetPaths":

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
        files = DatasetPaths(ds_dir=ds_dir, param_set_ids=param_set_ids, name=ds_name)

        files.validate()

        return files


@dataclass
class PetabDataset:

    name: str
    obs_df: pd.DataFrame
    cond_df: pd.DataFrame
    param_group: ParameterGroup
    meas_dfs: Dict[ParameterSetID, pd.DataFrame]

    files: Optional[DatasetPaths] = None

    def write(self, ds_dir: Directory) -> "PetabDataset":
        self.files = DatasetPaths(
            ds_dir=ds_dir,
            param_set_ids=list(self.meas_dfs.keys()),
            name=self.name,
        )

        self.files.make_dirs()

        pet.PetabIO.write_obs_df(self.obs_df, filename=self.files.obs_path)
        pet.PetabIO.write_cond_df(self.cond_df, filename=self.files.cond_path)
        for id, meas_df in self.meas_dfs.items():
            pet.PetabIO.write_meas_df(meas_df, filename=self.files.meas_paths_dict[id])

        self.param_group.write(self.files.tparam_path)

        return self

    @staticmethod
    def load(ds_dir: str) -> "PetabDataset":

        files = DatasetPaths.from_dir(ds_dir)

        obs_df = pet.PetabIO.read_obs_df(files.obs_path)
        cond_df = pet.PetabIO.read_cond_df(files.cond_path)
        param_group = ParameterGroup.load(files.tparam_path)
        meas_dfs = {
            id: pet.PetabIO.read_meas_df(meas_fp)
            for id, meas_fp in files.meas_paths_dict.items()
        }

        return PetabDataset(
            name=files.name,
            obs_df=obs_df,
            cond_df=cond_df,
            param_group=param_group,
            meas_dfs=meas_dfs,
            files=files,
        )
