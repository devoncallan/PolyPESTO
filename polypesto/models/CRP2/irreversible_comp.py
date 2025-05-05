from typing import Dict, Tuple
import pandas as pd
from numpy.typing import ArrayLike

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup, ParameterSet, Parameter, ParameterID

from polypesto.models import sbml, ModelInterface
from .common import define_irreversible_k


class IrreversibleComp(ModelInterface):
    """Irreversible Copolymerization Equation Model"""

    name: str = "irreversible_comp"

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        return irreversible_comp

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
        return IrreversibleComp.create_conditions(fA0=[0.5], cM0=[1.0])


def irreversible_comp() -> Tuple[sbml.Document, sbml.Model]:

    name = IrreversibleComp.name
    print(f"Creating SBML model: {name}")

    document, model = sbml.create_model(name)
    c = sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    (kpAA, kpAB, kpBA, kpBB) = define_irreversible_k(model)

    # Define initial concenetrations.
    A0 = sbml.create_parameter(model, "A0", value=1.0, units="mole", constant=True)
    B0 = sbml.create_parameter(model, "B0", value=1.0, units="mole", constant=True)

    A = sbml.create_parameter(model, "A", value=0, units="mole")
    B = sbml.create_parameter(model, "B", value=0, units="mole")
    fA = sbml.create_parameter(model, "fA", value=0, units="dimensionless")
    fB = sbml.create_parameter(model, "fB", value=0, units="dimensionless")

    sbml.create_rule(model, fA, formula=f"A / A0")
    sbml.create_rule(model, fB, formula=f"B / B0")

    FA = sbml.create_parameter(model, "FA", value=0, units="dimensionless")

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
