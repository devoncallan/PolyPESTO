from typing import Dict, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd

from polypesto.core import petab as pet
from . import sbml


AMICI_MODEL_DIR = Path(__file__).parent.parent / "amici_models"


class ModelBase(ABC):

    def __init__(
        self,
        data_dir: str,
        observables: Optional[List[str]] = None,
        obs_noise_level: float = 0.02,
    ):

        self.name = self.__class__.__name__
        self.data_dir = data_dir

        self.observables = {o: o for o in (observables or self._default_obs())}
        self.obs_noise_level = obs_noise_level

        self.fit_params = self._default_fit_params()

    @abstractmethod
    def _default_obs(self) -> List[str]:
        """Return default observables"""
        pass

    @abstractmethod
    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
        """Return default fit parameters"""
        pass

    @abstractmethod
    def sbml_model_def(self) -> sbml.ModelDefinition:
        """Return SBML model definition"""
        pass

    def get_param_df(self) -> pd.DataFrame:
        """Get fit parameter dataframe"""
        return pet.define_parameters(self.fit_params)

    def get_obs_df(self) -> pd.DataFrame:
        """Get observables dataframe"""
        return pet.define_observables(
            self.observables, noise_value=self.obs_noise_level
        )
