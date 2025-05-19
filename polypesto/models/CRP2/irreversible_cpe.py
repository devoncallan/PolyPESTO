from typing import Dict, Tuple
import pandas as pd
from numpy.typing import ArrayLike

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup, ParameterSet, Parameter, ParameterID

from polypesto.models import sbml, ModelInterface
from .common import define_irreversible_k


class IrreversibleCPE(ModelInterface):
    """Irreversible Copolymerization Equation Model"""

    name: str = "irreversible_cpe"

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        return irreversible_ode

    @staticmethod
    def create_conditions(fA0: ArrayLike, cM0: ArrayLike, **kwargs) -> pd.DataFrame:
        """
        Create conditions dataframe for the model.

        Parameters
        ----------
        fA0 : ArrayLike
            Feed fraction of monomer A
        cM0 : ArrayLike
            Total monomer concentration

        Returns
        -------
        pd.DataFrame
            Dataframe with initial conditions
        """
        import numpy as np

        # Convert to numpy arrays for element-wise multiplication
        fA0_array = np.array(fA0)
        cM0_array = np.array(cM0)

        return pet.define_conditions(
            {
                "A0": fA0_array * cM0_array,
                "B0": (1 - fA0_array) * cM0_array,
            }
        )

    @staticmethod
    def create_observables(**kwargs) -> pd.DataFrame:
        return pet.define_observables(**kwargs)

    @staticmethod
    def get_default_fit_params() -> Dict[str, pet.FitParameter]:
        """
        Get default fit parameters for the model.

        Returns
        -------
        Dict[str, pet.FitParameter]
            Dictionary of fit parameters
        """
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

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        """
        Get default parameter dataframe for fitting.

        Returns
        -------
        pd.DataFrame
            Parameter dataframe for PEtab
        """
        return pet.define_parameters(IrreversibleCPE.get_default_fit_params())

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        """
        Get default observables dataframe.

        Returns
        -------
        pd.DataFrame
            Observables dataframe for PEtab
        """
        return pet.define_observables({"xA": "xA", "xB": "xB"}, noise_value=0.02)

    @staticmethod
    def get_default_conditions() -> pd.DataFrame:
        """
        Get default conditions dataframe.

        Returns
        -------
        pd.DataFrame
            Conditions dataframe for PEtab
        """
        return IrreversibleCPE.create_conditions(fA0=[0.5], cM0=[1.0])


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

    # Define unreacted monomer composition
    fA = sbml.create_parameter(model, "fA", value=0)
    sbml.create_rule(model, fA, formula="A / (A + B + 1e-10)")
    fB = sbml.create_parameter(model, "fB", value=0)
    sbml.create_rule(model, fB, formula="1 - fA")

    # Define dxA/dt (dX)
    sbml.create_rate_rule(model, xA, formula="(A0+B0)/A0 * ((dA+1e-10)/(dA+dB+1e-10))")

    return document, model


def irreversible_ode() -> Tuple[sbml.Document, sbml.Model]:

    name = IrreversibleCPE.name
    print(f"Creating SBML model: {name}")

    document, model = sbml.create_model(name)
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    (kpAA, kpAB, kpBA, kpBB) = define_irreversible_k(model, kpAA_constant=True)
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
    sbml.create_rule(model, dA_dt, formula=f"-A*(kpAA*(R + PA) + kpBA*(R + PB))")
    sbml.create_rule(model, dB_dt, formula=f"-B*(kpBB*(R + PB) + kpAB*(R + PA))")

    # Define polymer balances
    sbml.create_rule(model, dPAA_dt, formula=f"kpAA*PA*A - PAA*(kpAA*A + kpAB*B)")
    sbml.create_rule(model, dPAB_dt, formula=f"kpAB*PA*B - PAB*(kpBA*A + kpBB*B)")
    sbml.create_rule(model, dPBA_dt, formula=f"kpBA*PB*A - PBA*(kpAB*B + kpAA*A)")
    sbml.create_rule(model, dPBB_dt, formula=f"kpBB*PB*B - PBB*(kpBB*B + kpBA*A)")

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
