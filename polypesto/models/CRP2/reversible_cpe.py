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
        return reversible_ode

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


def reversible_ode() -> Tuple[sbml.Document, sbml.Model]:

    name = ReversibleCPE.name
    print(f"Creating SBML model: {name}")

    document, model = sbml.create_model(name)
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    (kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB) = define_reversible_k(
        model, kpAA_constant=True
    )
    eps = sbml.create_parameter(model, "eps", value=1e-10, units="dimensionless")

    # Define all species
    A = sbml.create_parameter(model, "A", value=0)
    B = sbml.create_parameter(model, "B", value=0)

    R = sbml.create_species(model, "R", initialAmount=0.001)
    RA = sbml.create_species(model, "RA")
    RB = sbml.create_species(model, "RB")
    PAA = sbml.create_species(model, "PAA")
    PAB = sbml.create_species(model, "PAB")
    PBA = sbml.create_species(model, "PBA")
    PBB = sbml.create_species(model, "PBB")
    PA = sbml.create_species(model, "PA")
    PB = sbml.create_species(model, "PB")
    fPAA = sbml.create_parameter(model, "fPAA", value=0)
    fPAB = sbml.create_parameter(model, "fPAB", value=0)
    fPBA = sbml.create_parameter(model, "fPBA", value=0)
    fPBB = sbml.create_parameter(model, "fPBB", value=0)

    # Define initial concenetrations
    A0 = sbml.create_parameter(model, "A0", value=1.0, constant=True)
    B0 = sbml.create_parameter(model, "B0", value=1.0, constant=True)

    # Define observables – conversion and mole fraction
    xA = sbml.create_species(model, "xA", initialAmount=0.0)
    xB = sbml.create_species(model, "xB", initialAmount=0.0)
    fA = sbml.create_parameter(model, "fA", value=0)
    fB = sbml.create_parameter(model, "fB", value=0)

    # Define assignment rules
    sbml.create_rule(model, A, formula="A0 * (1 - xA)")
    sbml.create_rule(model, B, formula="(A0 + B0)*(1 - time) - A")

    sbml.create_rule(model, PA, formula="PAA + PBA + RA")
    sbml.create_rule(model, PB, formula="PAB + PBB + RB")

    sbml.create_rule(model, fA, formula="A / (A + B + eps)")
    sbml.create_rule(model, fB, formula="1 - fA")

    sbml.create_rule(model, fPAA, formula="PAA/(PA + eps)")
    sbml.create_rule(model, fPAB, formula="PAB/(PB + eps)")
    sbml.create_rule(model, fPBA, formula="PBA/(PA + eps)")
    sbml.create_rule(model, fPBB, formula="PBB/(PB + eps)")

    dR_dt = sbml.create_parameter(model, "dR_dt", value=0)
    dRA_dt = sbml.create_parameter(model, "dRA_dt", value=0)
    dRB_dt = sbml.create_parameter(model, "dRB_dt", value=0)
    dA_dt = sbml.create_parameter(model, "dA_dt", value=0)
    dB_dt = sbml.create_parameter(model, "dB_dt", value=0)
    dPAA_dt = sbml.create_parameter(model, "dPAA_dt", value=0)
    dPAB_dt = sbml.create_parameter(model, "dPAB_dt", value=0)
    dPBA_dt = sbml.create_parameter(model, "dPBA_dt", value=0)
    dPBB_dt = sbml.create_parameter(model, "dPBB_dt", value=0)
    dxA_dt = sbml.create_parameter(model, "dxA_dt", value=0)
    dxB_dt = sbml.create_parameter(model, "dxB_dt", value=0)
    dx_dt = sbml.create_parameter(model, "dx_dt", value=0)

    # Define R, RA, RB balances
    sbml.create_rule(model, dR_dt, formula=f"-R*(kpAA*A + kpBB*B)")
    sbml.create_rule(model, dRA_dt, formula=f"R*(kpAA*A) - RA*(kpAA*A + kpAB*B)")
    sbml.create_rule(model, dRB_dt, formula=f"R*(kpBB*B) - RB*(kpBB*B + kpBA*A)")

    # Define monomer balances
    sbml.create_rule(
        model,
        dA_dt,
        formula=f"-A*(kpAA*(R + PA) + kpBA*(R + PB)) + kdAA*PAA + kdBA*PBA",
    )
    sbml.create_rule(
        model,
        dB_dt,
        formula=f"-B*(kpBB*(R + PB) + kpAB*(R + PA)) + kdBB*PBB + kdAB*PAB",
    )

    # Define polymer balances
    sbml.create_rule(
        model,
        dPAA_dt,
        formula=f"kpAA*PA*A - PAA*(kpAA*A + kpAB*B) + kdAA*fPAA*PAA + kdAB*fPAA*PAB - kdAA*PAA",
    )
    sbml.create_rule(
        model,
        dPAB_dt,
        formula=f"kpAB*PA*B - PAB*(kpBA*A + kpBB*B) + kdBA*fPAB*PBA + kdBB*fPAB*PBB - kdAB*PAB",
    )
    sbml.create_rule(
        model,
        dPBA_dt,
        formula=f"kpBA*PB*A - PBA*(kpAB*B + kpAA*A) + kdAB*fPBA*PAB + kdAA*fPBA*PAA - kdBA*PBA",
    )
    sbml.create_rule(
        model,
        dPBB_dt,
        formula=f"kpBB*PB*B - PBB*(kpBB*B + kpBA*A) + kdBB*fPBB*PBB + kdBA*fPBB*PBA - kdBB*PBB",
    )

    # Define dx/dt (dx)
    sbml.create_rule(model, dxA_dt, formula="-1/A0 * dA_dt")
    sbml.create_rule(model, dxB_dt, formula="-1/B0 * dB_dt")
    sbml.create_rule(model, dx_dt, formula="-1/(A0+B0) * (dA_dt + dB_dt)")

    # Define dxA/dx (dX)
    sbml.create_rate_rule(model, xA, formula="dxA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, xB, formula="dxB_dt/(dx_dt+eps)")

    sbml.create_rate_rule(model, R, formula="dR_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, RA, formula="dRA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, RB, formula="dRB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, PAA, formula="dPAA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, PAB, formula="dPAB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, PBA, formula="dPBA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, PBB, formula="dPBB_dt/(dx_dt+eps)")

    return document, model


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

    fA = sbml.create_parameter(model, "fA", value=0)
    sbml.create_rule(model, fA, formula="A / (A + B + 1e-10)")
    fB = sbml.create_parameter(model, "fB", value=0)
    sbml.create_rule(model, fB, formula="1 - fA")

    # Irreversible: -B*(kpB*pB + kpAB*pA)
    # Reversible: -B*(kpBB*pB + kpAB*pA) + pB*(kdBB*pBB + kdAB*pAB)

    # Define dxA/dt (dX)
    sbml.create_rate_rule(model, xA, formula="(A0+B0)/A0 * ((dA+1e-10)/(dA+dB+1e-10))")

    return document, model
