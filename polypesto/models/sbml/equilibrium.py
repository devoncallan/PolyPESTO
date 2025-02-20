import libsbml
from .. import sbml

#############################
### Binary Reaction Model ###
#############################

# A -k1-> B
# B -k2-> A


def equilibrium_ode() -> sbml.ModelDefinition:
    print("Creating SBML model (Simple Chemical Equilibrium).")
    document, model = sbml.create_model()

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


def equilibrium_rxn_ode() -> sbml.ModelDefinition:
    print("Creating SBML model (Simple Chemical Equilibrium).")
    document, model = sbml.create_model()

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
