import libsbml
from .. import sbml
from typing import List

######################################################
### Controlled Radical Polymerization (CRP) models ###
######################################################


def define_irreversible_k(
    model: sbml.Model, kpAA_constant=False
) -> List[libsbml.Parameter]:

    print("Creating irreversible CRP parameters.")
    kpAA = sbml.create_parameter(model, "kpAA", value=1, constant=kpAA_constant)
    kpAB = sbml.create_parameter(model, "kpAB", value=1)
    kpBA = sbml.create_parameter(model, "kpBA", value=1)
    kpBB = sbml.create_parameter(model, "kpBB", value=1)

    rA = sbml.create_parameter(model, "rA", value=1)
    rB = sbml.create_parameter(model, "rB", value=1)
    rX = sbml.create_parameter(model, "rX", value=1)

    sbml.create_rule(model, kpAB, formula=f"{kpAA.getId()} / {rA.getId()}")
    sbml.create_rule(model, kpBB, formula=f"{kpAA.getId()} / {rX.getId()}")
    sbml.create_rule(model, kpBA, formula=f"{kpBB.getId()} / {rB.getId()}")

    return [kpAA, kpAB, kpBA, kpBB]


def define_reversible_k(model: sbml.Model, **kwargs) -> List[libsbml.Parameter]:

    [kpAA, kpAB, kpBA, kpBB] = define_irreversible_k(model, **kwargs)

    print("Creating reversile CRP parameters.")
    kdAA = sbml.create_parameter(model, "kdAA", value=1)
    kdAB = sbml.create_parameter(model, "kdAB", value=1)
    kdBA = sbml.create_parameter(model, "kdBA", value=1)
    kdBB = sbml.create_parameter(model, "kdBB", value=1)

    KAA = sbml.create_parameter(model, "KAA", value=0)
    KAB = sbml.create_parameter(model, "KAB", value=0)
    KBA = sbml.create_parameter(model, "KBA", value=0)
    KBB = sbml.create_parameter(model, "KBB", value=0)

    sbml.create_rule(model, kdAA, formula=f"{kpAA.getId()} * {KAA.getId()}")
    sbml.create_rule(model, kdAB, formula=f"{kpAB.getId()} * {KAB.getId()}")
    sbml.create_rule(model, kdBA, formula=f"{kpBA.getId()} * {KBA.getId()}")
    sbml.create_rule(model, kdBB, formula=f"{kpBB.getId()} * {KBB.getId()}")

    return [kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB]


#################################
### Copolymer Equation Models ###
#################################


def irreversible_cpe() -> sbml.ModelDefinition:

    print("Creating SBML model: irreversible_cpe")

    document, model = sbml.create_model()
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    (kpAA, kpAB, kpBA, kpBB) = define_irreversible_k(model)

    # Define initial concenetrations.
    A0 = sbml.create_parameter(model, "A0", value=1.0, units="mole", constant=True)
    B0 = sbml.create_parameter(model, "B0", value=1.0, units="mole", constant=True)

    # Define monomer concentration and conversion
    xA = sbml.create_species(model, "xA", initialAmount=0.0)
    A = sbml.create_parameter(model, "A", value=0, units="mole")
    sbml.create_rule(model, A, formula=f"A0 * (1 - xA)")

    xB = sbml.create_parameter(model, "xB", value=0)
    B = sbml.create_parameter(model, "B", value=0, units="mole")
    sbml.create_rule(model, B, formula=f"(A0 + B0)*(1 - time) - A")
    sbml.create_rule(model, xB, formula="1 - B / B0")

    # Define rates of change of monomer concentration
    dA = sbml.create_parameter(model, "dA", value=0)
    sbml.create_rule(model, dA, formula=f"-A*(rA*A + B)")

    dB = sbml.create_parameter(model, "dB", value=0)
    sbml.create_rule(model, dB, formula=f"-B*(A+rB*B)")

    # Define dxA/dt (dX)
    sbml.create_rate_rule(model, xA, formula="(A0+B0)/A0 * ((dA+1e-10)/(dA+dB+1e-10))")

    return document, model


def reversible_cpe() -> sbml.ModelDefinition:

    print("Creating SBML model: reversible_cpe")

    document, model = sbml.create_model()
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    (kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB) = define_reversible_k(
        model, kpAA_constant=True
    )

    # fA0, and M
    # xA = fA0*M

    # Define initial concenetrations.
    A0 = sbml.create_parameter(model, "A0", value=1.0, units="mole", constant=True)
    B0 = sbml.create_parameter(model, "B0", value=1.0, units="mole", constant=True)

    # Define monomer concentration and conversion
    xA = sbml.create_species(model, "xA", initialAmount=0.0)
    A = sbml.create_parameter(model, "A", value=0, units="mole")
    sbml.create_rule(model, A, formula=f"A0 * (1 - xA)")

    xB = sbml.create_parameter(model, "xB", value=0)
    B = sbml.create_parameter(model, "B", value=0, units="mole")
    sbml.create_rule(model, B, formula=f"(A0 + B0)*(1 - time) - A")
    sbml.create_rule(model, xB, formula="1 - B / B0")

    # sbml.create_initial_assignment(model, A0.getId(), formula="A")
    # sbml.create_initial_assignment(model, B0.getId(), formula="B")

    # Define terminal chain-end fractions
    pA = sbml.create_species(model, "pA", initialAmount=0.5)
    pB = sbml.create_species(model, "pB", initialAmount=0.5)

    # Define chain-end dyad fractions
    pAA = sbml.create_species(model, "pAA", initialAmount=0.5)
    pAB = sbml.create_species(model, "pAB", initialAmount=0.5)
    pBA = sbml.create_species(model, "pBA", initialAmount=0.5)
    pBB = sbml.create_species(model, "pBB", initialAmount=0.5)

    # Define chain-end triad balances
    sbml.create_algebraic_rule(
        model, formula="kpAA*pBA*pA*A + kdAB*pAA*pAB*pB - pAA*pA*(kpAB*B + kdAA*pBA)"
    )
    sbml.create_algebraic_rule(
        model, formula="kpBB*pAB*pB*B + kdBA*pBB*pBA*pA - pBB*pB*(kpBA*A + kdBB*pAB)"
    )
    sbml.create_algebraic_rule(
        model,
        formula="kpAB*pA*B + pAB*(kdBA*pBA*pA + kdBB*pBB*pB) - pAB*pB*(kpBA*A + kpBB*B + kdAB)",
    )

    # Identity rules
    sbml.create_rule(model, pB, formula="1 - pA")
    sbml.create_rule(model, pBB, formula="1 - pAB")
    sbml.create_rule(model, pBA, formula="1 - pAA")

    # Define rates of change of monomer concentration
    dA = sbml.create_parameter(model, "dA", value=0)
    sbml.create_rule(
        model, dA, formula=f"-A*(kpAA*pA + kpBA*pB) + pA*(kdAA*pAA + kdBA*pBA)"
    )

    dB = sbml.create_parameter(model, "dB", value=0)
    sbml.create_rule(
        model, dB, formula=f"-B*(kpBB*pB + kpAB*pA) + pB*(kdBB*pBB + kdAB*pAB)"
    )

    # Define dxA/dt (dX)
    sbml.create_rate_rule(model, xA, formula="(A0+B0)/A0 * ((dA+1e-10)/(dA+dB+1e-10))")

    is_valid_xA = sbml.create_parameter(model, "is_valid_xA", value=1)

    return document, model


#################################
### Reaction-Based ODE Models ###
#################################


def irreversible_ode() -> sbml.ModelDefinition:

    print(f"Creating SBML model: irreversible_ode")

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

    return document, model


def reversible_ode() -> sbml.ModelDefinition:

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
