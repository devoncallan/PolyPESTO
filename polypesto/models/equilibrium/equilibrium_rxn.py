from typing import Dict

import pandas as pd

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup
from polypesto.models import sbml, ModelInterface

from .equilibrium_ode import EquilibriumODE


#############################
### Binary Reaction Model ###
#############################

# A -k1-> B
# B -k2-> A


class EquilibriumRxn(ModelInterface):
    """Equilibrium Rxn (Reaction) Model"""
    
    name: str = "equilibrium_rxn"

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        return equilibrium_rxn

    @staticmethod
    def create_conditions(A, B) -> pd.DataFrame:
        return EquilibriumODE.create_conditions(A, B)

    @staticmethod
    def get_default_fit_params() -> Dict[str, pet.FitParameter]:
        return EquilibriumODE.get_default_fit_params()

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        return EquilibriumRxn.get_default_parameters()

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        return EquilibriumODE.get_default_observables()

    @staticmethod
    def get_default_conditions() -> pd.DataFrame:
        return EquilibriumODE.get_default_conditions()


def equilibrium_rxn() -> sbml.ModelDefinition:
    
    name = EquilibriumRxn.name
    print(f"Creating SBML model ({name}).")
    document, model = sbml.create_model(name)

    # Create compartment (0D since we're dealing with concentrations)
    c = sbml.create_compartment(model, "c", spatialDimensions=3)

    # Define rate constants (k1 for A->B, k2 for B->A)
    k1 = sbml.create_parameter(model, "k1", value=0.5)
    k2 = sbml.create_parameter(model, "k2", value=0.5)

    A = sbml.create_species(model, "A", initialAmount=1.0)
    B = sbml.create_species(model, "B", initialAmount=0.0)

    # Define reactions
    # Syntax: (reaction_id, {reactants: stoich}, {products: stoich}, kinetic_law)
    reactions = [
        # First propagation
        (
            "forward_rxn",
            {"A": 1},
            {"B": 1},
            "k1*A",
        ),
        (
            "backward_rxn",
            {"B": 1},
            {"A": 1},
            "k2*B",
        ),
    ]

    print("Creating reactions.")
    generated_reactions = []
    for r in reactions:  # (reaction_id, reactants_dict, products_dict, kinetic_law)
        reaction = sbml.create_reaction(model, r[0], r[1], r[2], r[3])
        generated_reactions.append(reaction)

    return document, model
