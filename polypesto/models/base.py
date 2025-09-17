from typing import Dict, List, Optional, Callable
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd
from amici.amici import AmiciSolver

from polypesto.core import petab as pet
from . import sbml


AMICI_MODEL_DIR = Path(__file__).parent.parent / "amici_models"


class ModelBase(ABC):

    def __init__(
        self,
        observables: Optional[List[str]] = None,
        obs_noise_level: float = 0.02,
        sbml_model: Optional[sbml.ModelDefinition] = None,
        solver_options: Optional[Callable[[AmiciSolver], AmiciSolver]] = None,
    ):

        self.name = self.__class__.__name__

        self.observables = {o: o for o in (observables or self._default_obs())}
        self.obs_noise_level = obs_noise_level

        self.fit_params = self._default_fit_params()
        self.sbml_model = sbml_model if sbml_model else self._default_sbml_model()

        if solver_options is None:
            self.solver_options = self._default_solver_options
        elif not callable(solver_options):
            raise TypeError(
                "solver_options must be a function that takes and returns an AmiciSolver."
            )
        else:
            self.solver_options = solver_options

    @abstractmethod
    def _default_obs(self) -> List[str]:
        """Return default observables"""
        pass

    @abstractmethod
    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
        """Return default fit parameters"""
        pass

    @abstractmethod
    def _default_sbml_model(self) -> sbml.ModelDefinition:
        """Return default sbml model."""
        pass

    @abstractmethod
    def _default_solver_options(self, solver: AmiciSolver) -> AmiciSolver:
        """Default solver options"""
        pass

    def get_param_df(self) -> pd.DataFrame:
        """Get fit parameter dataframe"""
        return pet.define_parameters(self.fit_params)

    def get_obs_df(self) -> pd.DataFrame:
        """Get observables dataframe"""
        return pet.define_observables(
            self.observables, noise_value=self.obs_noise_level
        )

    def model_name_with_hash(self) -> str:
        """
        Get a unique model name based on its observables and fit parameters.

        Returns:
            str: A unique model name.
        """

        import hashlib

        obs_str = str(sorted(self.observables.keys()))
        obs_hash_str = hashlib.md5(obs_str.encode()).hexdigest()

        # Fit parameter fields that affect model compilation
        fit_signature = {
            param_id: (param.estimate, param.scale)
            for param_id, param in self.fit_params.items()
        }
        fit_str = str(sorted(fit_signature.items()))
        fit_hash_str = hashlib.md5(fit_str.encode()).hexdigest()

        combined_str = f"{obs_hash_str}_{fit_hash_str}"
        combined_hash_str = hashlib.md5(combined_str.encode()).hexdigest()

        return f"{self.name}_{combined_hash_str}"
