from typing import Dict, List

from polypesto.core import petab as pet
from polypesto.models import sbml, ModelBase


#############################
### Binary Reaction Model ###
#############################

# A -k1-> B
# B -k2-> A


class Equilibrium(ModelBase):
    """Simple Equilibrium Reaction Model"""

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

    def _default_sbml_model(self) -> sbml.ModelDefinition:
        return equilibrium_ode()


def equilibrium_ode() -> sbml.ModelDefinition:

    document, model = sbml.init_model("equilibrium_ode")

    # Create compartment (0D since we're dealing with concentrations)
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    # Define rate constants (k1 for A->B, k2 for B->A)
    sbml.create_parameter(model, "k1", value=0.5)
    sbml.create_parameter(model, "k2", value=0.5)

    sbml.create_species(model, "A", initialAmount=1.0)
    sbml.create_species(model, "B", initialAmount=0.0)

    # dx/dt = k2*y - k1*x
    sbml.create_rate_rule(model, "A", formula="k2*B - k1*A")

    # dy/dt = k1*x - k2*y
    sbml.create_rate_rule(model, "B", formula="k1*A - k2*B")

    return sbml.create_model(model, document)


def equilibrium_rxn() -> sbml.ModelDefinition:

    document, model = sbml.init_model("equilibrium_rxn")

    # Create compartment (0D since we're dealing with concentrations)
    sbml.create_compartment(model, "c", spatialDimensions=3)

    # Define rate constants (k1 for A->B, k2 for B->A)
    sbml.create_parameter(model, "k1", value=0.5)
    sbml.create_parameter(model, "k2", value=0.5)

    sbml.create_species(model, "A", initialAmount=1.0)
    sbml.create_species(model, "B", initialAmount=0.0)

    # Define reactions
    # Syntax: (reaction_id, {reactants: stoich}, {products: stoich}, kinetic_law)
    reactions = [
        # First propagation
        ("forward_rxn", {"A": 1}, {"B": 1}, "k1*A"),
        ("backward_rxn", {"B": 1}, {"A": 1}, "k2*B"),
    ]

    for r in reactions:  # (reaction_id, reactants_dict, products_dict, kinetic_law)
        sbml.create_reaction(model, r[0], r[1], r[2], r[3])

    return sbml.create_model(model, document)
