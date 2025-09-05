from typing import Dict, List

from polypesto.core import petab as pet
from polypesto.models import sbml, ModelBase


#############################
### Binary Reaction Model ###
#############################


# A -k1-> B
# B -k2-> A
class EquilibriumODE(ModelBase):
    """Equilibrium ODE (Reaction) Model"""

    def _default_obs(self) -> List[str]:
        return ["xA", "xB"]

    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
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

    def sbml_model_def(self) -> sbml.ModelDefinition:

        return equilibrium_ode


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
