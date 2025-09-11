from typing import Dict, List

from polypesto.core import petab as pet
from polypesto.models import sbml, ModelBase
from .common import define_reversible_k


class ReversibleCPE(ModelBase):

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
            "KAA": pet.FitParameter(
                id="KAA",
                scale=pet.C.LOG10,
                bounds=(1e-2, 1e2),
                nominal_value=1.0,
                estimate=False,
            ),
            "KAB": pet.FitParameter(
                id="KAB",
                scale=pet.C.LIN,
                bounds=(0, 1),
                nominal_value=0.0,
                estimate=False,
            ),
            "KBA": pet.FitParameter(
                id="KBA",
                scale=pet.C.LIN,
                bounds=(0, 1),
                nominal_value=0.0,
                estimate=False,
            ),
            "KBB": pet.FitParameter(
                id="KBB",
                scale=pet.C.LIN,
                bounds=(0, 1),
                nominal_value=0.0,
                estimate=False,
            ),
        }

    def _default_sbml_model(self) -> sbml.ModelDefinition:
        return reversible_ode()


def reversible_ode() -> sbml.ModelDefinition:

    document, model = sbml.init_model("reversible_cpe (ode)")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    define_reversible_k(model, kpAA_constant=True)
    sbml.create_parameter(model, "eps", value=1e-10, units="dimensionless")

    # Define all species
    sbml.create_parameter(model, "A")
    sbml.create_parameter(model, "B")
    sbml.create_species(model, "R", initialAmount=0.001)
    sbml.create_species(model, "RA")
    sbml.create_species(model, "RB")
    sbml.create_species(model, "PAA")
    sbml.create_species(model, "PAB")
    sbml.create_species(model, "PBA")
    sbml.create_species(model, "PBB")
    sbml.create_species(model, "PA")
    sbml.create_species(model, "PB")
    sbml.create_parameter(model, "fPAA")
    sbml.create_parameter(model, "fPAB")
    sbml.create_parameter(model, "fPBA")
    sbml.create_parameter(model, "fPBB")

    # Define initial concenetrations
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

    sbml.create_rule(model, "fPAA", formula="PAA/(PA + eps)")
    sbml.create_rule(model, "fPAB", formula="PAB/(PB + eps)")
    sbml.create_rule(model, "fPBA", formula="PBA/(PA + eps)")
    sbml.create_rule(model, "fPBB", formula="PBB/(PB + eps)")

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
    sbml.create_rule(
        model,
        "dA_dt",
        formula=f"-A*(kpAA*(R + PA) + kpBA*(R + PB)) + kdAA*PAA + kdBA*PBA",
    )
    sbml.create_rule(
        model,
        "dB_dt",
        formula=f"-B*(kpBB*(R + PB) + kpAB*(R + PA)) + kdBB*PBB + kdAB*PAB",
    )

    # Define polymer balances
    sbml.create_rule(
        model,
        "dPAA_dt",
        formula=f"kpAA*PA*A - PAA*(kpAA*A + kpAB*B) + kdAA*fPAA*PAA + kdAB*fPAA*PAB - kdAA*PAA",
    )
    sbml.create_rule(
        model,
        "dPAB_dt",
        formula=f"kpAB*PA*B - PAB*(kpBA*A + kpBB*B) + kdBA*fPAB*PBA + kdBB*fPAB*PBB - kdAB*PAB",
    )
    sbml.create_rule(
        model,
        "dPBA_dt",
        formula=f"kpBA*PB*A - PBA*(kpAB*B + kpAA*A) + kdAB*fPBA*PAB + kdAA*fPBA*PAA - kdBA*PBA",
    )
    sbml.create_rule(
        model,
        "dPBB_dt",
        formula=f"kpBB*PB*B - PBB*(kpBB*B + kpBA*A) + kdBB*fPBB*PBB + kdBA*fPBB*PBA - kdBB*PBB",
    )

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


def reversible_cpe() -> sbml.ModelDefinition:

    document, model = sbml.init_model("reversible_cpe (cpe)")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    define_reversible_k(model, kpAA_constant=True)

    # Define initial concentrations.
    sbml.create_species(model, "A0", initialAmount=1.0)
    sbml.create_species(model, "B0", initialAmount=1.0)

    # Define monomer concentration and conversion
    sbml.create_species(model, "xA", initialAmount=0.0)
    sbml.create_parameter(model, "A", value=0, units="mole")
    sbml.create_rule(model, "A", formula=f"A0 * (1 - xA)")

    sbml.create_parameter(model, "xB", value=0)
    sbml.create_parameter(model, "B", value=0, units="mole")
    sbml.create_rule(model, "B", formula=f"(A0 + B0)*(1 - time) - A")
    sbml.create_rule(model, "xB", formula="1 - B / B0")

    # Define terminal chain-end fractions
    sbml.create_species(model, "pA", initialAmount=0.5)
    sbml.create_species(model, "pB", initialAmount=0.5)

    # Define chain-end dyad fractions
    sbml.create_species(model, "pAA", initialAmount=0.5)
    sbml.create_species(model, "pAB", initialAmount=0.5)
    sbml.create_species(model, "pBA", initialAmount=0.5)
    sbml.create_species(model, "pBB", initialAmount=0.5)

    # Define chain-end triad balances
    sbml.create_algebraic_rule(
        model, formula="kpAA*pBA*pA*A + kdAB*pAA*pAB*pB - pAA*pA*(kpAB*B + kdAA*pBA)"
    )
    # Irreversible: dpAA/dt = 0 = kpAA*pBA*pA*A - kpAB*pAA*pA*B
    # Reversible: dpAA/dt = 0 = kpAA*pBA*pA*A + kdAB*pAA*pAB*pB - pAA*pA*(kpAB*B + kdAA*pBA)

    sbml.create_algebraic_rule(
        model, formula="kpBB*pAB*pB*B + kdBA*pBB*pBA*pA - pBB*pB*(kpBA*A + kdBB*pAB)"
    )
    # Irreversible: dpBB/dt = 0 = kpBB*pAB*pB*B - kpBA*pBB*pB*A
    # Reversible: dpBB/dt = 0 = kpBB*pAB*pB*B + kdBA*pBB*pBA*pA - pBB*pB*(kpBA*A + kdBB*pAB)

    # sbml.create_algebraic_rule(
    #     model,
    #     formula="kpAB*pA*B + pAB*(kdBA*pBA*pA + kdBB*pBB*pB) - pAB*pB*(kpBA*A + kpBB*B + kdAB)",
    # )
    # Irreversible: dPAB/dt = 0 = kpAB*pA*B - kpBA*pAB*pB*A - kpBB*pAB*pB*B
    # Reversible: dPAB/dt = 0 = kpAB*pA*B + pAB*(kdBA*pBA*pA + kdBB*pBB*pB) - pAB*pB*(kpBA*A + kpBB*B + kdAB)
    sbml.create_algebraic_rule(
        model, formula="kpAB*pA*B + kdBA*pBA*pA - kpBA*pB*A - kdAB*pAB*pB"
    )

    # Identity rules
    sbml.create_rule(model, "pA", formula="1 - pB")
    sbml.create_rule(model, "pAA", formula="1 - pBA")
    sbml.create_rule(model, "pBB", formula="1 - pAB")

    # Define rates of change of monomer concentration
    sbml.create_parameter(model, "dA", value=0)
    sbml.create_rule(
        model, "dA", formula=f"-A*(kpAA*pA + kpBA*pB) + pA*(kdAA*pAA + kdBA*pBA)"
    )
    # Irreversible: -A*(kpA*pA + kpBA*pB)
    # Reversible: -A*(kpAA*pA + kpBA*pB) + pA*(kdAA*pAA + kdBA*pBA)

    sbml.create_parameter(model, "dB", value=0)
    sbml.create_rule(
        model, "dB", formula=f"-B*(kpBB*pB + kpAB*pA) + pB*(kdBB*pBB + kdAB*pAB)"
    )

    sbml.create_parameter(model, "fA", value=0)
    sbml.create_rule(model, "fA", formula="A / (A + B + 1e-10)")
    sbml.create_parameter(model, "fB", value=0)
    sbml.create_rule(model, "fB", formula="1 - fA")

    # Irreversible: -B*(kpB*pB + kpAB*pA)
    # Reversible: -B*(kpBB*pB + kpAB*pA) + pB*(kdBB*pBB + kdAB*pAB)

    # Define dxA/dt (dX)
    sbml.create_rate_rule(
        model, "xA", formula="(A0+B0)/A0 * ((dA+1e-10)/(dA+dB+1e-10))"
    )

    return sbml.create_model(model, document)


def reversible_rxn() -> sbml.ModelDefinition:

    document, model = sbml.init_model("reversible_rxn")
    sbml.create_compartment(model, "c")

    define_reversible_k(model)

    sbml.create_species(model, "R", initialAmount=0.001)

    sbml.create_parameter(model, "A0")
    sbml.create_parameter(model, "B0")

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
    sbml.create_parameter(model, "xA", value=0)
    sbml.create_parameter(model, "xB", value=0)
    sbml.create_rule(model, "xA", formula=f"1 - A/A0")
    sbml.create_rule(model, "xB", formula=f"1 - B/B0")

    # Define chain-end dyad fractions
    sbml.create_parameter(model, "fPAA", value=1)
    sbml.create_parameter(model, "fPAB", value=1)
    sbml.create_parameter(model, "fPBA", value=1)
    sbml.create_parameter(model, "fPBB", value=1)
    sbml.create_rule(model, "PA", formula=f"PAA + PBA")
    sbml.create_rule(model, "PB", formula=f"PAB + PBB")

    eps = 1e-10
    sbml.create_rule(model, "fPAA", formula=f"(PAA + {eps}) / (PA + {eps})")
    sbml.create_rule(model, "fPAB", formula=f"(PAB + {eps}) / (PB + {eps})")
    sbml.create_rule(model, "fPBA", formula=f"(PBA + {eps}) / (PA + {eps})")
    sbml.create_rule(model, "fPBB", formula=f"(PBB + {eps}) / (PB + {eps})")

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
        # Depropagation
        ("deprop_PAAA", {"PAA": 1}, {"PAA": 1, "A": 1}, "kdAA * fPAA * PAA"),
        ("deprop_PBAA", {"PAA": 1}, {"PBA": 1, "A": 1}, "kdAA * fPBA * PAA"),
        ("deprop_PABA", {"PBA": 1}, {"PAB": 1, "A": 1}, "kdBA * fPAB * PBA"),
        ("deprop_PBBA", {"PBA": 1}, {"PBB": 1, "A": 1}, "kdBA * fPBB * PBA"),
        ("deprop_PAAB", {"PAB": 1}, {"PAA": 1, "B": 1}, "kdAB * fPAA * PAB"),
        ("deprop_PBAB", {"PAB": 1}, {"PBA": 1, "B": 1}, "kdAB * fPBA * PAB"),
        ("deprop_PABB", {"PBB": 1}, {"PAB": 1, "B": 1}, "kdBB * fPAB * PBB"),
        ("deprop_PBBB", {"PBB": 1}, {"PBB": 1, "B": 1}, "kdBB * fPBB * PBB"),
    ]

    for r in reactions:  # (reaction_id, reactants_dict, products_dict, kinetic_law)
        sbml.create_reaction(model, r[0], r[1], r[2], r[3])

    return sbml.create_model(model, document)
