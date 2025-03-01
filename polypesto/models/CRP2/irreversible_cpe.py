from typing import Dict, Tuple
import pandas as pd

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup, ParameterSet, Parameter, ParameterID

from polypesto.models import sbml, ModelInterface
from .common import define_irreversible_k


class IrreversibleCPE(ModelInterface):
    """Irreversible Copolymerization Equation Model"""

    name: str = "irreversible_cpe"

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        return irreversible_cpe

    @staticmethod
    def create_conditions(fA0s, cM0s) -> pd.DataFrame:
        return pet.define_conditions(
            {
                "A0": fA0s * cM0s,
                "B0": (1 - fA0s) * cM0s,
            }
        )

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
                bounds=(1e-2, 1e2),
                nominal_value=1.0,
                estimate=False,
            ),
        }

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        return pet.define_parameters(IrreversibleCPE.default_fit_params())

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        return pet.define_observables({"xA": "xA", "xB": "xB"}, noise_value=0.02)

    @staticmethod
    def get_default_conditions() -> pd.DataFrame:
        return IrreversibleCPE.create_conditions_df([1], [1])


def irreversible_cpe() -> Tuple[sbml.Document, sbml.Model]:

    name = IrreversibleCPE.name
    print(f"Creating SBML model: {name}")

    document, model = sbml.create_model(name)
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
