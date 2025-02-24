# Define Object

# Define SBML model
from typing import Dict
import pandas as pd

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup

from polypesto.models import sbml
from .common import define_reversible_k
from .reversible_cpe import ReversibleCPE


class ReversibleRxn(sbml.ModelInterface):
    """Reversible Copolymerization ODE (Reaction) Model"""

    @staticmethod
    def create_sbml_model() -> sbml.ModelDefinition:
        return reversible_rxn()

    @staticmethod
    def create_conditions(fA0s, cM0s) -> pd.DataFrame:
        return ReversibleCPE.create_conditions(fA0s, cM0s)

    @staticmethod
    def get_default_fit_params() -> Dict[str, pet.FitParameter]:
        return ReversibleCPE.get_default_fit_params()

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        return pet.define_parameters(ReversibleCPE.default_fit_params())

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        return ReversibleCPE.get_default_observables()

    @staticmethod
    def get_default_conditions() -> pd.DataFrame:
        return ReversibleCPE.create_conditions([1], [1])

    @staticmethod
    def get_simulation_parameters() -> ParameterGroup:
        return ReversibleCPE.get_simulation_parameters()


def reversible_rxn() -> sbml.ModelDefinition:

    print(f"Creating SBML model reversible_ode")

    document, model = sbml.create_model()
    c = sbml.create_compartment(model, "c")

    print("Creating species.")
    R = sbml.create_species(model, "R", initialAmount=0.001)

    A = sbml.create_species(model, "A", initialAmount=0.5)
    B = sbml.create_species(model, "B", initialAmount=0.5)

    RA = sbml.create_species(model, "RA")
    RB = sbml.create_species(model, "RB")

    PAA = sbml.create_species(model, "PAA")
    PAB = sbml.create_species(model, "PAB")
    PBA = sbml.create_species(model, "PBA")
    PBB = sbml.create_species(model, "PBB")

    PA = sbml.create_species(model, "PA")
    PB = sbml.create_species(model, "PB")

    (kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB) = define_reversible_k(model)

    # Calculates monomer conversion
    A0 = sbml.create_parameter(model, "A0", value=0)
    B0 = sbml.create_parameter(model, "B0", value=0)
    xA = sbml.create_parameter(model, "xA", value=0)
    xB = sbml.create_parameter(model, "xB", value=0)

    sbml.create_initial_assignment(model, A0.getId(), formula=f"{A.getId()}")
    sbml.create_initial_assignment(model, B0.getId(), formula=f"{B.getId()}")

    sbml.create_rule(model, xA, formula=f"1 - {A.getId()}/{A0.getId()}")
    sbml.create_rule(model, xB, formula=f"1 - {B.getId()}/{B0.getId()}")

    # Define chain-end dyad fractions
    fPAA = sbml.create_parameter(model, "fPAA", value=1)
    fPAB = sbml.create_parameter(model, "fPAB", value=1)
    fPBA = sbml.create_parameter(model, "fPBA", value=1)
    fPBB = sbml.create_parameter(model, "fPBB", value=1)
    sbml.create_rule(model, PA, formula=f"{PAA.getId()} + {PBA.getId()}")
    sbml.create_rule(model, PB, formula=f"{PAB.getId()} + {PBB.getId()}")

    eps = 1e-10
    sbml.create_rule(
        model, fPAA, formula=f"({PAA.getId()} + {eps}) / ({PA.getId()} + {eps})"
    )
    sbml.create_rule(
        model, fPAB, formula=f"({PAB.getId()} + {eps}) / ({PB.getId()} + {eps})"
    )
    sbml.create_rule(
        model, fPBA, formula=f"({PBA.getId()} + {eps}) / ({PA.getId()} + {eps})"
    )
    sbml.create_rule(
        model, fPBB, formula=f"({PBB.getId()} + {eps}) / ({PB.getId()} + {eps})"
    )

    # Defining reactions
    # Syntax: (reaction_id, {reactants: stoich}, {products: stoich}, kinetic_law)
    reactions = [
        # First propagation
        (
            "prop_A",
            {R.getId(): 1, A.getId(): 1},
            {RA.getId(): 1},
            f"{kpAA.getId()} * {R.getId()} * {A.getId()}",
        ),
        (
            "prop_B",
            {R.getId(): 1, B.getId(): 1},
            {RB.getId(): 1},
            f"{kpBB.getId()} * {R.getId()} * {B.getId()}",
        ),
        (
            "prop_RA_A",
            {RA.getId(): 1, A.getId(): 1},
            {PAA.getId(): 1},
            f"{kpAA.getId()} * {RA.getId()} * {A.getId()}",
        ),
        (
            "prop_RA_B",
            {RA.getId(): 1, B.getId(): 1},
            {PAB.getId(): 1},
            f"{kpAB.getId()} * {RA.getId()} * {B.getId()}",
        ),
        (
            "prop_RB_A",
            {RB.getId(): 1, A.getId(): 1},
            {PBA.getId(): 1},
            f"{kpBA.getId()} * {RB.getId()} * {A.getId()}",
        ),
        (
            "prop_RB_B",
            {RB.getId(): 1, B.getId(): 1},
            {PBB.getId(): 1},
            f"{kpBB.getId()} * {RB.getId()} * {B.getId()}",
        ),
        # Propagation
        (
            "prop_PAA_A",
            {PAA.getId(): 1, A.getId(): 1},
            {PAA.getId(): 1},
            f"{kpAA.getId()} * {PAA.getId()} * {A.getId()}",
        ),
        (
            "prop_PAA_B",
            {PAA.getId(): 1, B.getId(): 1},
            {PAB.getId(): 1},
            f"{kpAB.getId()} * {PAA.getId()} * {B.getId()}",
        ),
        (
            "prop_PAB_A",
            {PAB.getId(): 1, A.getId(): 1},
            {PBA.getId(): 1},
            f"{kpBA.getId()} * {PAB.getId()} * {A.getId()}",
        ),
        (
            "prop_PAB_B",
            {PAB.getId(): 1, B.getId(): 1},
            {PBB.getId(): 1},
            f"{kpBB.getId()} * {PAB.getId()} * {B.getId()}",
        ),
        (
            "prop_PBA_A",
            {PBA.getId(): 1, A.getId(): 1},
            {PAA.getId(): 1},
            f"{kpAA.getId()} * {PBA.getId()} * {A.getId()}",
        ),
        (
            "prop_PBA_B",
            {PBA.getId(): 1, B.getId(): 1},
            {PAB.getId(): 1},
            f"{kpAB.getId()} * {PBA.getId()} * {B.getId()}",
        ),
        (
            "prop_PBB_A",
            {PBB.getId(): 1, A.getId(): 1},
            {PBA.getId(): 1},
            f"{kpBA.getId()} * {PBB.getId()} * {A.getId()}",
        ),
        (
            "prop_PBB_B",
            {PBB.getId(): 1, B.getId(): 1},
            {PBB.getId(): 1},
            f"{kpBB.getId()} * {PBB.getId()} * {B.getId()}",
        ),
        # Depropagation
        (
            "deprop_PAAA",
            {PAA.getId(): 1},
            {PAA.getId(): 1, A.getId(): 1},
            f"{kdAA.getId()} * {fPAA.getId()} * {PAA.getId()}",
        ),
        (
            "deprop_PBAA",
            {PAA.getId(): 1},
            {PBA.getId(): 1, A.getId(): 1},
            f"{kdAA.getId()} * {fPBA.getId()} * {PAA.getId()}",
        ),
        (
            "deprop_PABA",
            {PBA.getId(): 1},
            {PAB.getId(): 1, A.getId(): 1},
            f"{kdBA.getId()} * {fPAB.getId()} * {PBA.getId()}",
        ),
        (
            "deprop_PBBA",
            {PBA.getId(): 1},
            {PBB.getId(): 1, A.getId(): 1},
            f"{kdBA.getId()} * {fPBB.getId()} * {PBA.getId()}",
        ),
        (
            "deprop_PAAB",
            {PAB.getId(): 1},
            {PAA.getId(): 1, B.getId(): 1},
            f"{kdAB.getId()} * {fPAA.getId()} * {PAB.getId()}",
        ),
        (
            "deprop_PBAB",
            {PAB.getId(): 1},
            {PBA.getId(): 1, B.getId(): 1},
            f"{kdAB.getId()} * {fPBA.getId()} * {PAB.getId()}",
        ),
        (
            "deprop_PABB",
            {PBB.getId(): 1},
            {PAB.getId(): 1, B.getId(): 1},
            f"{kdBB.getId()} * {fPAB.getId()} * {PBB.getId()}",
        ),
        (
            "deprop_PBBB",
            {PBB.getId(): 1},
            {PBB.getId(): 1, B.getId(): 1},
            f"{kdBB.getId()} * {fPBB.getId()} * {PBB.getId()}",
        ),
    ]

    print("Creating reactions.")
    generated_reactions = []
    for r in reactions:  # (reaction_id, reactants_dict, products_dict, kinetic_law)
        reaction = sbml.create_reaction(model, r[0], r[1], r[2], r[3])
        generated_reactions.append(reaction)

    return document, model
