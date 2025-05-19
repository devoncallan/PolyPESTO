from typing import Dict, Tuple
import numpy as np
import pandas as pd

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup
from polypesto.models import sbml, ModelInterface

from .common import define_irreversible_k
from .irreversible_cpe import IrreversibleCPE


class IrreversibleRxn(ModelInterface):
    """Irreversible Copolymerization ODE (Reaction) Model"""

    name: str = "irreversible_rxn"

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        return irreversible_rxn

    @staticmethod
    def create_conditions(fA0, cM0) -> pd.DataFrame:

        fA0 = np.array(fA0)
        cM0 = np.array(cM0)

        A0 = fA0 * cM0
        B0 = (1 - fA0) * cM0

        return pet.define_conditions(
            {
                "A0": A0,
                "B0": B0,
            }
        )

    @staticmethod
    def create_observables(**kwargs) -> pd.DataFrame:
        return pet.define_observables(**kwargs)

    @staticmethod
    def get_default_fit_params() -> Dict[str, pet.FitParameter]:
        return IrreversibleCPE.get_default_fit_params()

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        return IrreversibleCPE.get_default_parameters()

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        return IrreversibleCPE.get_default_observables()

    @staticmethod
    def get_default_conditions() -> pd.DataFrame:
        return IrreversibleCPE.get_default_conditions()


def irreversible_rxn() -> Tuple[sbml.Document, sbml.Model]:

    name = IrreversibleRxn.name
    print(f"Creating SBML model: {name}")

    document, model = sbml.create_model(name)
    c = sbml.create_compartment(model, "c")

    print("Creating species.")
    R = sbml.create_species(model, "R", initialAmount=0.001)

    A0 = sbml.create_species(model, "A0")
    B0 = sbml.create_species(model, "B0")

    A = sbml.create_species(model, "A")
    B = sbml.create_species(model, "B")

    sbml.create_initial_assignment(model, "A", formula="A0")
    sbml.create_initial_assignment(model, "B", formula="B0")

    RA = sbml.create_species(model, "RA")
    RB = sbml.create_species(model, "RB")

    PAA = sbml.create_species(model, "PAA")
    PAB = sbml.create_species(model, "PAB")
    PBA = sbml.create_species(model, "PBA")
    PBB = sbml.create_species(model, "PBB")

    PA = sbml.create_species(model, "PA")
    PB = sbml.create_species(model, "PB")

    (kpAA, kpAB, kpBA, kpBB) = define_irreversible_k(model)

    # Calculates monomer conversion

    xA = sbml.create_species(model, "xA")
    xB = sbml.create_species(model, "xB")
    x = sbml.create_species(model, "x")
    fA = sbml.create_species(model, "fA")
    fB = sbml.create_species(model, "fB")

    sbml.create_rule(model, xA, formula=f"1 - A/A0")
    sbml.create_rule(model, xB, formula=f"1 - B/B0")
    sbml.create_rule(model, x, formula=f"1 - (A0+B0-A-B)/(A0+B0+1e-10)")
    sbml.create_rule(model, fA, formula=f"A/(A+B+1e-10)")
    sbml.create_rule(model, fB, formula=f"B/(A+B+1e-10)")

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
    ]

    print("Creating reactions.")
    generated_reactions = []
    for r in reactions:  # (reaction_id, reactants_dict, products_dict, kinetic_law)
        reaction = sbml.create_reaction(model, r[0], r[1], r[2], r[3])
        generated_reactions.append(reaction)

    # x_thresholds = np.arange(0.01, 1, 0.01)
    # print(x_thresholds)
    # sbml.add_conversion_snapshot_events(model, x_thresholds=x_thresholds)
    return document, model
