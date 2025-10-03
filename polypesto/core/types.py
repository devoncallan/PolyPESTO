from typing import TypeAlias

# ==========================================
# Type aliases for SBML-related types
# ==========================================

import libsbml
from petab.v1.models.sbml_model import SbmlModel  # type: ignore

ModelDefinition: TypeAlias = SbmlModel
Document: TypeAlias = libsbml.SBMLDocument
Model: TypeAlias = libsbml.Model


# ==========================================
# Type aliases for PyPESTO-related types
# ==========================================

from pypesto import Problem as PypestoProblem  # type: ignore
from pypesto import Result  # type: ignore
from pypesto.ensemble import Ensemble  # type: ignore
from pypesto.ensemble import EnsemblePrediction  # type: ignore
from pypesto.petab import PetabImporter  # type: ignore
from pypesto.objective import AmiciObjective  # type: ignore

