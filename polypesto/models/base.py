from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from amici.amici import AmiciSolver  # type: ignore

from polypesto.core import petab as pet
from polypesto.utils import ID

from . import sbml
from .utils import parse_obs_noise

AMICI_MODEL_DIR = Path(__file__).parent.parent / "amici_models"


class ModelBase(ABC):

    def __init__(
        self,
        observables: List[ID.StrObsName] | None = None,
        obs_noise: float | List[float] | Dict[ID.StrObsName, float] | None = None,
        sbml_model: sbml.ModelDefinition | None = None,
        solver_options: Callable[[AmiciSolver], AmiciSolver] | None = None,
    ):

        self.name = self.__class__.__name__

        self.obs_names = observables or self._default_obs()
        self.obs_formula_map = {o: o for o in self.obs_names}
        self.obs_noise_map = parse_obs_noise(obs_noise, self.obs_names)

        self.fit_params = self._default_fit_params()
        self.sbml_model = sbml_model if sbml_model else self._default_sbml_model()

        if solver_options is None or not callable(solver_options):
            self.solver_options = self._default_solver_options
        else:
            # Type assertion to help mypy understand the type after callable() check
            self.solver_options = solver_options  # type: ignore[assignment]

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

    def _default_solver_options(self, solver: AmiciSolver) -> AmiciSolver:
        """Default solver options"""
        print("Using default solver options...")
        # solver.setNewtonMaxSteps(10_000)
        # solver.setNewtonDampingFactorMode(1)
        # solver.setAbsoluteTolerance(1e-10)
        # solver.setRelativeTolerance(1e-6)
        # solver.setMaxSteps(10_000)
        # solver.setMaxConvFails(1_000)
        # solver.setMaxNonlinIters(10_000)
        # solver.setLinearSolver(9)
        # solver.setStabilityLimitFlag(True)
        # solver.setReturnDataReportingMode(0)
        # solver.setLinearMultistepMethod(2)
        return solver

    def get_param_df(self) -> pd.DataFrame:
        """Get fit parameter dataframe"""
        return pet.utils.param.define(self.fit_params)

    def get_obs_df(self) -> pd.DataFrame:
        """Get observables dataframe"""
        return pet.utils.obs.define(self.obs_formula_map, self.obs_noise_map)

    def model_name_with_hash(self) -> str:
        """
        Get a unique model name based on its observables and fit parameters.

        Returns:
            str: A unique model name.
        """

        sbml_str = str(self.sbml_model.model_id)
        sbml_hash_str = ID.get_hash(sbml_str)

        obs_str = str(sorted(self.obs_names))
        obs_hash_str = ID.get_hash(obs_str)

        # Fit parameter fields that affect model compilation
        fit_signature = {
            param_id: (param.estimate, param.scale)
            for param_id, param in self.fit_params.items()
        }
        fit_str = str(sorted(fit_signature.items()))
        fit_hash_str = ID.get_hash(fit_str)

        combined_str = f"{sbml_hash_str}_{obs_hash_str}_{fit_hash_str}"
        combined_hash_str = ID.get_hash(combined_str)

        return f"{self.name}_{combined_hash_str}"
