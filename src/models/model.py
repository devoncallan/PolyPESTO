from abc import ABC, abstractmethod
from src.utils.params import ParameterSet, ParameterGroup, ParameterSetID
from src.petab.dataset import PetabDataset
from typing import Any, Dict, List, Optional
import pandas as pd


class Model(ABC):

    def __init__(
        self,
        name: str,
        model: Any,
        obs_df: pd.DataFrame,
        model_filepath: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.model = model
        self.obs_df = obs_df
        self.model_filepath = model_filepath

    def get_obs_df(self):
        if self.obs_df is None:
            raise ValueError(f"Observation data is not available for {self.name}")
        return self.obs_df

    def get_cond_df(self):
        if self.cond_df is None:
            raise ValueError(f"Condition data is not available for {self.name}")
        return self.cond_df

    @abstractmethod
    def set_params(self, param_set: ParameterSet):
        pass

    @abstractmethod
    def simulate(
        self,
        t_eval: List[float],
        conditions: Dict[str, float],
        cond_id: str = None,
        **kwargs,
    ) -> pd.DataFrame:
        pass

    def generate_dataset(
        self,
        param_group: ParameterGroup,
        t_eval: List[float],
        cond_df: pd.DataFrame,
        name: Optional[str] = None,
        **kwargs,
    ) -> PetabDataset:

        meas_dfs: Dict[ParameterSetID, pd.DataFrame] = {}

        for param_set in param_group.get_parameter_sets():

            self.set_params(param_set)

            meas_dfs_temp = []
            for cond_id, conditions in cond_df.iterrows():
                conditions = conditions.to_dict()

                meas_df = self.simulate(t_eval, conditions, cond_id=cond_id, **kwargs)
                meas_dfs_temp.append(meas_df)

            meas_df = pd.concat(meas_dfs_temp)
            meas_dfs[param_set.id] = meas_df

        return PetabDataset(
            name=name,
            obs_df=self.get_obs_df(),
            cond_df=cond_df,
            param_group=param_group,
            meas_dfs=meas_dfs,
            model_filepath=self.model_filepath,
        )
