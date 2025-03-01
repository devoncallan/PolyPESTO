from typing import Protocol, Dict, Optional
from dataclasses import dataclass
import pandas as pd

from polypesto.core.petab import (
    PetabData,
    FitParameter,
    define_empty_measurements,
    define_parameters,
)
from . import sbml


class ModelInterface(Protocol):

    name: str

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        """Create and return SBML model"""
        ...

    @staticmethod
    def create_conditions(**kwargs) -> pd.DataFrame:
        """Create conditions dataframe"""
        ...

    @staticmethod
    def create_default_conditions() -> pd.DataFrame:
        """Create default conditions dataframe"""
        ...

    @staticmethod
    def get_default_fit_params():
        """Return default parameter settings"""
        ...

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        """Return default parameters dataframe"""
        ...

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        """Return default observables dataframe"""
        ...

    @staticmethod
    def get_simulation_parameters():
        """Return simulation parameters"""
        ...

    # @classmethod
    # def create_petab_data(
    #     cls, conditions: Dict, params_dict: Optional[Dict[str, FitParameter]] = None
    # ) -> PetabData:
    #     """Create PEtab data from experiment conditions"""
    #     if params_dict is None:
    #         params_dict = cls.get_default_fit_params()

    #     param_df = define_parameters(params_dict)
    #     cond_df = cls.create_conditions(**conditions)
    #     obs_df = cls.get_default_observables()
    #     empty_meas_df = define_empty_measurements(obs_df, cond_df, conditions["t_eval"])

    #     return PetabData(
    #         obs_df=obs_df, cond_df=cond_df, param_df=param_df, meas_df=empty_meas_df
    #     )
