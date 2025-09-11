from typing import Dict, List

from polypesto.core import petab as pet
from polypesto.models import sbml, ModelBase
from .common import define_irreversible_k


class IrreversibleCPE(ModelBase):

    def _default_obs(self) -> List[str]:
        return ["xA", "xB"]

    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
        return {
            "rA": pet.FitParameter(
                id="rA",
                scale=pet.C.LOG10,
                bounds=(1e-3, 1e2),
                nominal_value=1.0,
                estimate=True,
            ),
            "rB": pet.FitParameter(
                id="rB",
                scale=pet.C.LOG10,
                bounds=(1e-3, 1e2),
                nominal_value=1.0,
                estimate=True,
            ),
            "rX": pet.FitParameter(
                id="rX",
                scale=pet.C.LOG10,
                bounds=(1e-2, 1e2),
                nominal_value=1.0,
                estimate=False,
            ),
        }

    def _default_sbml_model(self) -> sbml.ModelDefinition:
        return irreversible_ode()


def irreversible_cpe() -> sbml.ModelDefinition:

    document, model = sbml.init_model("irreversible_cpe")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    define_irreversible_k(model)

    # Define initial concenetrations.
    sbml.create_parameter(model, "A0", value=1.0, units="mole", constant=True)
    sbml.create_parameter(model, "B0", value=1.0, units="mole", constant=True)

    # Define monomer concentration and conversion
    sbml.create_species(model, "xA", initialAmount=0.0)
    sbml.create_parameter(model, "A", value=0, units="mole")
    sbml.create_rule(model, "A", formula=f"A0 * (1 - xA)")

    sbml.create_parameter(model, "xB", value=0)
    sbml.create_parameter(model, "B", value=0, units="mole")
    sbml.create_rule(model, "B", formula=f"(A0 + B0)*(1 - time) - A")
    sbml.create_rule(model, "xB", formula="1 - B / B0")

    # Define rates of change of monomer concentration
    sbml.create_parameter(model, "dA", value=0)
    sbml.create_rule(model, "dA", formula=f"-A*(rA*A + B)")

    sbml.create_parameter(model, "dB", value=0)
    sbml.create_rule(model, "dB", formula=f"-B*(A+rB*B)")

    # Define unreacted monomer composition
    sbml.create_parameter(model, "fA", value=0)
    sbml.create_rule(model, "fA", formula="A / (A + B + 1e-10)")
    sbml.create_parameter(model, "fB", value=0)
    sbml.create_rule(model, "fB", formula="1 - fA")

    # Define dxA/dt (dX)
    sbml.create_rate_rule(
        model, "xA", formula="(A0+B0)/A0 * ((dA+1e-10)/(dA+dB+1e-10))"
    )

    return sbml.create_model(model, document)


def irreversible_ode() -> sbml.ModelDefinition:

    document, model = sbml.init_model("irreversible_cpe (ODE)")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    define_irreversible_k(model, kpAA_constant=True)
    sbml.create_parameter(model, "eps", value=1e-10, units="dimensionless")

    # Define all species
    sbml.create_parameter(model, "A", value=0)
    sbml.create_parameter(model, "B", value=0)

    sbml.create_species(model, "R", initialAmount=0.001)
    sbml.create_species(model, "RA")
    sbml.create_species(model, "RB")
    sbml.create_species(model, "PAA")
    sbml.create_species(model, "PAB")
    sbml.create_species(model, "PBA")
    sbml.create_species(model, "PBB")
    sbml.create_species(model, "PA")
    sbml.create_species(model, "PB")

    # Define initial concentrations
    sbml.create_parameter(model, "A0", value=1.0, constant=True)
    sbml.create_parameter(model, "B0", value=1.0, constant=True)

    # Define observables – conversion and mole fraction
    sbml.create_species(model, "xA", initialAmount=0.0)
    sbml.create_species(model, "xB", initialAmount=0.0)
    sbml.create_parameter(model, "fA", value=0)
    sbml.create_parameter(model, "fB", value=0)

    # Define assignment rules
    sbml.create_rule(model, "A", formula="A0 * (1 - xA)")
    sbml.create_rule(model, "B", formula="(A0 + B0)*(1 - time) - A")

    sbml.create_rule(model, "PA", formula="PAA + PBA + RA")
    sbml.create_rule(model, "PB", formula="PAB + PBB + RB")

    sbml.create_rule(model, "fA", formula="A / (A + B + eps)")
    sbml.create_rule(model, "fB", formula="1 - fA")

    sbml.create_parameter(model, "dR_dt", value=0)
    sbml.create_parameter(model, "dRA_dt", value=0)
    sbml.create_parameter(model, "dRB_dt", value=0)
    sbml.create_parameter(model, "dA_dt", value=0)
    sbml.create_parameter(model, "dB_dt", value=0)
    sbml.create_parameter(model, "dPAA_dt", value=0)
    sbml.create_parameter(model, "dPAB_dt", value=0)
    sbml.create_parameter(model, "dPBA_dt", value=0)
    sbml.create_parameter(model, "dPBB_dt", value=0)
    sbml.create_parameter(model, "dxA_dt", value=0)
    sbml.create_parameter(model, "dxB_dt", value=0)
    sbml.create_parameter(model, "dx_dt", value=0)

    # Define R, RA, RB balances
    sbml.create_rule(model, "dR_dt", formula=f"-R*(kpAA*A + kpBB*B)")
    sbml.create_rule(model, "dRA_dt", formula=f"R*(kpAA*A) - RA*(kpAA*A + kpAB*B)")
    sbml.create_rule(model, "dRB_dt", formula=f"R*(kpBB*B) - RB*(kpBB*B + kpBA*A)")

    # Define monomer balances
    sbml.create_rule(model, "dA_dt", formula=f"-A*(kpAA*(R + PA) + kpBA*(R + PB))")
    sbml.create_rule(model, "dB_dt", formula=f"-B*(kpBB*(R + PB) + kpAB*(R + PA))")

    # Define polymer balances
    sbml.create_rule(model, "dPAA_dt", formula=f"kpAA*PA*A - PAA*(kpAA*A + kpAB*B)")
    sbml.create_rule(model, "dPAB_dt", formula=f"kpAB*PA*B - PAB*(kpBA*A + kpBB*B)")
    sbml.create_rule(model, "dPBA_dt", formula=f"kpBA*PB*A - PBA*(kpAB*B + kpAA*A)")
    sbml.create_rule(model, "dPBB_dt", formula=f"kpBB*PB*B - PBB*(kpBB*B + kpBA*A)")

    # Define dx/dt (dx)
    sbml.create_rule(model, "dxA_dt", formula="-1/A0 * dA_dt")
    sbml.create_rule(model, "dxB_dt", formula="-1/B0 * dB_dt")
    sbml.create_rule(model, "dx_dt", formula="-1/(A0+B0) * (dA_dt + dB_dt)")

    # Define dxA/dx (dX)
    sbml.create_rate_rule(model, "xA", formula="dxA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "xB", formula="dxB_dt/(dx_dt+eps)")

    sbml.create_rate_rule(model, "R", formula="dR_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "RA", formula="dRA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "RB", formula="dRB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PAA", formula="dPAA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PAB", formula="dPAB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PBA", formula="dPBA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PBB", formula="dPBB_dt/(dx_dt+eps)")

    return sbml.create_model(model, document)


def irreversible_rxn() -> sbml.ModelDefinition:

    document, model = sbml.init_model("irreversible_rxn")
    sbml.create_compartment(model, "c")

    define_irreversible_k(model)

    sbml.create_species(model, "R", initialAmount=0.001)

    sbml.create_species(model, "A0")
    sbml.create_species(model, "B0")

    sbml.create_species(model, "A")
    sbml.create_species(model, "B")

    sbml.create_initial_assignment(model, "A", formula="A0")
    sbml.create_initial_assignment(model, "B", formula="B0")

    sbml.create_species(model, "RA")
    sbml.create_species(model, "RB")

    sbml.create_species(model, "PAA")
    sbml.create_species(model, "PAB")
    sbml.create_species(model, "PBA")
    sbml.create_species(model, "PBB")

    sbml.create_species(model, "PA")
    sbml.create_species(model, "PB")

    # Calculates monomer conversion
    sbml.create_species(model, "xA")
    sbml.create_species(model, "xB")
    sbml.create_species(model, "x")
    sbml.create_species(model, "fA")
    sbml.create_species(model, "fB")

    sbml.create_rule(model, "xA", formula=f"1 - A/A0")
    sbml.create_rule(model, "xB", formula=f"1 - B/B0")
    sbml.create_rule(model, "x", formula=f"1 - (A0+B0-A-B)/(A0+B0+1e-10)")
    sbml.create_rule(model, "fA", formula=f"A/(A+B+1e-10)")
    sbml.create_rule(model, "fB", formula=f"B/(A+B+1e-10)")

    # Defining reactions
    # Syntax: (reaction_id, {reactants: stoich}, {products: stoich}, kinetic_law)
    reactions = [
        # First propagation
        ("prop_A", {"R": 1, "A": 1}, {"RA": 1}, "kpAA * R * A"),
        ("prop_B", {"R": 1, "B": 1}, {"RB": 1}, "kpBB * R * B"),
        ("prop_RA_A", {"RA": 1, "A": 1}, {"PAA": 1}, "kpAA * RA * A"),
        ("prop_RA_B", {"RA": 1, "B": 1}, {"PAB": 1}, "kpAB * RA * B"),
        ("prop_RB_A", {"RB": 1, "A": 1}, {"PBA": 1}, "kpBA * RB * A"),
        ("prop_RB_B", {"RB": 1, "B": 1}, {"PBB": 1}, "kpBB * RB * B"),
        # Propagation
        ("prop_PAA_A", {"PAA": 1, "A": 1}, {"PAA": 1}, "kpAA * PAA * A"),
        ("prop_PAA_B", {"PAA": 1, "B": 1}, {"PAB": 1}, "kpAB * PAA * B"),
        ("prop_PAB_A", {"PAB": 1, "A": 1}, {"PBA": 1}, "kpBA * PAB * A"),
        ("prop_PAB_B", {"PAB": 1, "B": 1}, {"PBB": 1}, "kpBB * PAB * B"),
        ("prop_PBA_A", {"PBA": 1, "A": 1}, {"PAA": 1}, "kpAA * PBA * A"),
        ("prop_PBA_B", {"PBA": 1, "B": 1}, {"PAB": 1}, "kpAB * PBA * B"),
        ("prop_PBB_A", {"PBB": 1, "A": 1}, {"PBA": 1}, "kpBA * PBB * A"),
        ("prop_PBB_B", {"PBB": 1, "B": 1}, {"PBB": 1}, "kpBB * PBB * B"),
    ]

    for r in reactions:  # (reaction_id, reactants_dict, products_dict, kinetic_law)
        sbml.create_reaction(model, r[0], r[1], r[2], r[3])

    return sbml.create_model(model, document)
