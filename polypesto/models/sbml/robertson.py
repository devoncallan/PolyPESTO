import libsbml
from .. import sbml
from typing import List

#######################
### Robertson Model ###
#######################

# A -k1-> B
# B + B -k2-> B + C
# B + C -k3-> A + C


def define_rob_k(model: sbml.Model) -> List[libsbml.Parameter]:

    print("Creating Robertson parameters.")
    k1 = sbml.create_parameter(model, "k1", value=0.04)
    k2 = sbml.create_parameter(model, "k2", value=1.0e4)
    k3 = sbml.create_parameter(model, "k3", value=3.0e7)

    return [k1, k2, k3]


def define_rob_species(model: sbml.Model) -> List[libsbml.Species]:

    print("Creating Robertson species.")
    A = sbml.create_species(model, "A", initialAmount=1.0)
    B = sbml.create_species(model, "B", initialAmount=0.0)
    C = sbml.create_species(model, "C", initialAmount=0.0)

    return [A, B, C]


def robertson_dae() -> sbml.ModelDefinition:

    print("Creating SBML model: robertson_dae")

    document, model = sbml.create_model()
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    # Define rate constants
    (k1, k2, k3) = define_rob_k(model)

    # Define species
    (A, B, C) = define_rob_species(model)

    # Define rate rules
    sbml.create_rate_rule(model, A, formula=f"-k1*A + k3*B*C")
    sbml.create_rate_rule(model, B, formula=f"k1*A - k3*B*C - k2*B*B")
    sbml.create_algebraic_rule(model, formula=f"1 - A - B - C")

    return document, model


def robertson_ode() -> sbml.ModelDefinition:

    print("Creating SBML model: robertson_ode")

    document, model = sbml.create_model()
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    # Define rate constants
    (k1, k2, k3) = define_rob_k(model)

    # Define species
    (A, B, C) = define_rob_species(model)

    # Define reactions
    # Syntax: (reaction_id, {reactants: stoich}, {products: stoich}, kinetic_law)
    reactions = [
        # First propagation
        (
            "r1",
            {"A": 1},
            {"B": 1},
            "k1*A",
        ),
        (
            "r2",
            {"B": 2},
            {"B": 1, "C": 1},
            "k2*B*B",
        ),
        (
            "r3",
            {"B": 1, "C": 1},
            {"A": 1, "C": 1},
            "k3*B*C",
        ),
    ]

    print("Creating reactions.")
    generated_reactions = []
    for r in reactions:  # (reaction_id, reactants_dict, products_dict, kinetic_law)
        reaction = sbml.create_reaction(model, r[0], r[1], r[2], r[3])
        generated_reactions.append(reaction)

    return document, model
