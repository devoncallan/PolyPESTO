from typing import Dict

import pandas as pd

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup
from polypesto.models import sbml, ModelInterface


#############################
### Binary Reaction Model ###
#############################

# A -k1-> B
# B -k2-> A


class EquilibriumODE(ModelInterface):
    """Equilibrium ODE (Reaction) Model"""

    name: str = "equilibrium_ode"

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        return equilibrium_ode

    @staticmethod
    def create_conditions(A, B) -> pd.DataFrame:
        return pet.define_conditions(
            {
                "A": A,
                "B": B,
            }
        )

    @staticmethod
    def get_default_fit_params() -> Dict[str, pet.FitParameter]:
        bounds = (1e-3, 1e3)
        return {
            "k1": pet.FitParameter(
                id="k1",
                scale=pet.C.LOG10,
                bounds=bounds,
                nominal_value=0.5,
                estimate=True,
            ),
            "k2": pet.FitParameter(
                id="k2",
                scale=pet.C.LOG10,
                bounds=bounds,
                nominal_value=0.5,
                estimate=True,
            ),
        }

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        return pet.define_parameters(EquilibriumODE.default_fit_params())

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        return pet.define_observables({"A": "A", "B": "B"}, noise_value=0.02)

    @staticmethod
    def get_default_conditions() -> pd.DataFrame:
        return EquilibriumODE.create_conditions([1], [1])


def equilibrium_ode() -> sbml.ModelDefinition:

    name = EquilibriumODE.name
    print(f"Creating SBML model ({name}).")
    document, model = sbml.create_model(name)

    # Create compartment (0D since we're dealing with concentrations)
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    # Define rate constants (k1 for A->B, k2 for B->A)
    k1 = sbml.create_parameter(model, "k1", value=0.5)
    k2 = sbml.create_parameter(model, "k2", value=0.5)

    A = sbml.create_species(model, "A", initialAmount=1.0)
    B = sbml.create_species(model, "B", initialAmount=0.0)

    # dx/dt = k2*y - k1*x
    sbml.create_rate_rule(model, A, formula="k2*B - k1*A")

    # dy/dt = k1*x - k2*y
    sbml.create_rate_rule(model, B, formula="k1*A - k2*B")

    return document, model
