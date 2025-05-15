from typing import Dict, Tuple
import pandas as pd
import numpy as np

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup, ParameterSet, Parameter, ParameterID

from polypesto.models import sbml, ModelInterface
from .common import define_reversible_k
from .irreversible_cpe import IrreversibleCPE


class ReversibleCPE(ModelInterface):
    """Irreversible Copolymerization Equation Model"""

    name: str = "reversible_cpe"

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        return reversible_cpe

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
        return {
            "rA": pet.FitParameter(
                id="rA",
                scale=pet.C.LOG10,
                bounds=(1e-2, 1e2),
                nominal_value=1.0,
                estimate=True,
            ),
            "rB": pet.FitParameter(
                id="rB",
                scale=pet.C.LOG10,
                bounds=(1e-2, 1e2),
                nominal_value=1.0,
                estimate=True,
            ),
            "rX": pet.FitParameter(
                id="rX",
                scale=pet.C.LOG10,
                bounds=(1e-3, 1e3),
                nominal_value=1.0,
                estimate=False,
            ),
            "KAA": pet.FitParameter(
                id="KAA",
                # scale=pet.C.LIN,
                # bounds=(0, 1),
                # nominal_value=0.0,
                # estimate=True,
                scale=pet.C.LOG10,
                bounds=(1e-2, 1e2),
                nominal_value=1.0,
                estimate=True,
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

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        return pet.define_parameters(ReversibleCPE.get_default_fit_params())

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        return IrreversibleCPE.get_default_observables()

    @staticmethod
    def get_default_conditions() -> pd.DataFrame:
        return IrreversibleCPE.create_conditions([1], [1])


def reversible_cpe() -> Tuple[sbml.Document, sbml.Model]:

    name = ReversibleCPE.name
    print(f"Creating SBML model: {name}")

    document, model = sbml.create_model(name)
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    (kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB) = define_reversible_k(
        model, kpAA_constant=True
    )

    # Define initial concenetrations.
    A0 = sbml.create_species(model, "A0", initialAmount=1.0)
    B0 = sbml.create_species(model, "B0", initialAmount=1.0)

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
    # Irreversible: dpAA/dt = 0 = kpAA*pBA*pA*A - kpAB*pAA*pA*B
    # Reversible: dpAA/dt = 0 = kpAA*pBA*pA*A + kdAB*pAA*pAB*pB - pAA*pA*(kpAB*B + kdAA*pBA)

    sbml.create_algebraic_rule(
        model, formula="kpBB*pAB*pB*B + kdBA*pBB*pBA*pA - pBB*pB*(kpBA*A + kdBB*pAB)"
    )
    # Irreversible: dpBB/dt = 0 = kpBB*pAB*pB*B - kpBA*pBB*pB*A
    # Reversible: dpBB/dt = 0 = kpBB*pAB*pB*B + kdBA*pBB*pBA*pA - pBB*pB*(kpBA*A + kdBB*pAB)

    sbml.create_algebraic_rule(
        model,
        formula="kpAB*pA*B + pAB*(kdBA*pBA*pA + kdBB*pBB*pB) - pAB*pB*(kpBA*A + kpBB*B + kdAB)",
    )
    # Irreversible: dPAB/dt = 0 = kpAB*pA*B - kpBA*pAB*pB*A - kpBB*pAB*pB*B
    # Reversible: dPAB/dt = 0 = kpAB*pA*B + pAB*(kdBA*pBA*pA + kdBB*pBB*pB) - pAB*pB*(kpBA*A + kpBB*B + kdAB)

    # Identity rules
    sbml.create_rule(model, pA, formula="1 - pB")
    sbml.create_rule(model, pAA, formula="1 - pBA")
    sbml.create_rule(model, pBB, formula="1 - pAB")
    # sbml.create_rule(model, pB, formula="1 - pA")
    # sbml.create_rule(model, pBB, formula="1 - pAB")
    # sbml.create_rule(model, pBA, formula="1 - pAA")

    # Define rates of change of monomer concentration
    dA = sbml.create_parameter(model, "dA", value=0)
    sbml.create_rule(
        model, dA, formula=f"-A*(kpAA*pA + kpBA*pB) + pA*(kdAA*pAA + kdBA*pBA)"
    )
    # Irreversible: -A*(kpA*pA + kpBA*pB)
    # Reversible: -A*(kpAA*pA + kpBA*pB) + pA*(kdAA*pAA + kdBA*pBA)

    dB = sbml.create_parameter(model, "dB", value=0)
    sbml.create_rule(
        model, dB, formula=f"-B*(kpBB*pB + kpAB*pA) + pB*(kdBB*pBB + kdAB*pAB)"
    )
    # Irreversible: -B*(kpB*pB + kpAB*pA)
    # Reversible: -B*(kpBB*pB + kpAB*pA) + pB*(kdBB*pBB + kdAB*pAB)

    # Define dxA/dt (dX)
    sbml.create_rate_rule(model, xA, formula="(A0+B0)/A0 * ((dA+1e-10)/(dA+dB+1e-10))")

    # Define dA/dB =

    # is_valid_xA = sbml.create_parameter(model, "is_valid_xA", value=1)

    return document, model


def irr_reversible_cpe() -> Tuple[sbml.Document, sbml.Model]:

    name = ReversibleCPE.name
    print(f"Creating SBML model: {name} (irreversible)")

    document, model = sbml.create_model(name)
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    (kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB) = define_reversible_k(
        model, kpAA_constant=True
    )

    # Define initial concenetrations.
    A0 = sbml.create_species(model, "A0", initialAmount=1.0)
    B0 = sbml.create_species(model, "B0", initialAmount=1.0)

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
    f_A = 0.9
    pA = sbml.create_species(model, "pA", initialAmount=0.5)
    pB = sbml.create_species(model, "pB", initialAmount=0.5)

    # Define chain-end dyad fractions
    pAA = sbml.create_species(model, "pAA", initialAmount=0.5)
    pAB = sbml.create_species(model, "pAB", initialAmount=0.5)
    pBA = sbml.create_species(model, "pBA", initialAmount=0.5)
    pBB = sbml.create_species(model, "pBB", initialAmount=0.5)

    print("h i")
    # Define chain-end triad balances
    sbml.create_algebraic_rule(model, formula="kpAA*pBA*pA*(A) - kpAB*pAA*pA*(B)")

    sbml.create_algebraic_rule(model, formula="kpBB*pAB*pB*(B) - kpBA*pBB*pB*(A)")

    sbml.create_algebraic_rule(
        model,
        formula="kpAB*pA*(B) - kpBA*pAB*pB*(A) - kpBB*pAB*pB*(B)",
    )

    # Identity rules
    print("hi.")
    sbml.create_rule(model, pA, formula="1 - pB")
    sbml.create_rule(model, pAA, formula="1 - pBA")
    sbml.create_rule(model, pBB, formula="1 - pAB")

    # Define rates of change of monomer concentration
    dA = sbml.create_parameter(model, "dA", value=0)
    sbml.create_rule(model, dA, formula=f"-A*(kpAA*pA + kpBA*pB)")

    dB = sbml.create_parameter(model, "dB", value=0)
    sbml.create_rule(model, dB, formula=f"-B*(kpBB*pB + kpAB*pA)")

    # Define dxA/dt (dX)
    sbml.create_rate_rule(model, xA, formula="(A0+B0)/A0 * ((dA+1e-10)/(dA+dB+1e-10))")

    print("hi")
    return document, model


def irr_reversible_cpe_() -> Tuple[sbml.Document, sbml.Model]:
    name = ReversibleCPE.name
    print(f"Creating SBML model: {name} (irreversible)")
    document, model = sbml.create_model(name)
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    # Define rate constants
    (kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB) = define_reversible_k(
        model, kpAA_constant=True
    )

    # Define initial concentrations
    A0 = sbml.create_species(model, "A0", initialAmount=1.0)
    B0 = sbml.create_species(model, "B0", initialAmount=1.0)

    # Define monomer concentration and conversion
    xA = sbml.create_species(model, "xA", initialAmount=0.0)
    A = sbml.create_parameter(model, "A", value=0, units="mole")
    sbml.create_rule(model, A, formula=f"A0 * (1 - xA)")

    xB = sbml.create_parameter(model, "xB", value=0)
    B = sbml.create_parameter(model, "B", value=0, units="mole")
    sbml.create_rule(model, B, formula=f"(A0 + B0)*(1 - time) - A")
    sbml.create_rule(model, xB, formula="1 - B / B0")

    # Add relaxation factor that decreases rapidly with conversion
    # relax = sbml.create_parameter(model, "relax", value=0, constant=True)
    # sbml.create_rule(model, relax, formula="1e-4*exp(-10 * time)")

    # Define terminal and penultimate chain-end fractions with improved initial values
    pA = sbml.create_species(model, "pA", initialAmount=0.5)
    pB = sbml.create_species(model, "pB", initialAmount=0.5)
    pAA = sbml.create_species(model, "pAA", initialAmount=0.5)
    pAB = sbml.create_species(model, "pAB", initialAmount=0.5)
    pBA = sbml.create_species(model, "pBA", initialAmount=0.5)
    pBB = sbml.create_species(model, "pBB", initialAmount=0.5)

    # Define algebraic rules with relaxation terms
    sbml.create_algebraic_rule(
        model,
        formula="kpAA*pBA*pA*(A) - kpAB*pAA*pA*(B)",
    )
    sbml.create_algebraic_rule(
        model,
        formula="kpBB*pAB*pB*(B) - kpBA*pBB*pB*(A)",
    )
    sbml.create_algebraic_rule(
        model,
        formula="kpAB*pA*(B) - kpBA*pAB*pB*(A) - kpBB*pAB*pB*(B)",
    )

    # Identity rules
    sbml.create_rule(model, pB, formula="1 - pA")
    sbml.create_rule(model, pBA, formula="1 - pAA")
    sbml.create_rule(model, pBB, formula="1 - pAB")

    # Define rates of change of monomer concentration
    dA = sbml.create_parameter(model, "dA", value=0)
    sbml.create_rule(model, dA, formula=f"-A*(kpAA*pA + kpBA*pB)")
    dB = sbml.create_parameter(model, "dB", value=0)
    sbml.create_rule(model, dB, formula=f"-B*(kpBB*pB + kpAB*pA)")

    fA = sbml.create_parameter(model, "fA", value=0)
    sbml.create_rule(model, fA, formula="A / (A + B + 1e-10)")
    fB = sbml.create_parameter(model, "fB", value=0)
    sbml.create_rule(model, fB, formula="1 - fA")

    # Define dxA/dt with regularization to prevent extreme values
    sbml.create_rate_rule(
        model,
        xA,
        formula="(A0+B0)/A0 * ((dA+1e-8)/(dA+dB+1e-8))",
    )

    return document, model
