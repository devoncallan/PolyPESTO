from typing import Protocol

import pandas as pd

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
    def create_observables(**kwargs) -> pd.DataFrame:
        """Create observables dataframe"""
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
