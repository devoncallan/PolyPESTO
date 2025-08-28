from typing import Dict

import pandas as pd

from polypesto.core import petab as pet
from polypesto.core.params import ParameterGroup
from polypesto.models import sbml, ModelInterface


#############################
### Lotka-Volterra Equations ###
#############################

# dx/dt = a*x - b*x*y
# dy/dt = -c*y + d*x*y


class LotkaVolterraODE(ModelInterface):
    """Lotka Volterra ODE Model"""

    name: str = "lotka_volterra_ode"

    @staticmethod
    def sbml_model_def() -> sbml.ModelDefinition:
        return lotka_volterra_ode

    @staticmethod
    def create_conditions(x, y) -> pd.DataFrame:
        return pet.define_conditions(
            {
                "x": x,
                "y": y,
            }
        )

    @staticmethod
    def get_default_fit_params() -> Dict[str, pet.FitParameter]:
        return {
            "a": pet.FitParameter(
                id="a",
                scale=pet.C.LIN,
                bounds=(1e-6, 10),
                nominal_value=1.0,
                estimate=True,
            ),
            "b": pet.FitParameter(
                id="b",
                scale=pet.C.LIN,
                bounds=(1e-8, 1),
                nominal_value=0.1,
                estimate=True,
            ),
            "c": pet.FitParameter(
                id="c",
                scale=pet.C.LIN,
                bounds=(1e-6, 1),
                nominal_value=0.075,
                estimate=True,
            ),
            "d": pet.FitParameter(
                id="d",
                scale=pet.C.LIN,
                bounds=(1e-6, 10),
                nominal_value=1.5,
                estimate=True,
            ),
        }

    @staticmethod
    def get_default_parameters() -> pd.DataFrame:
        return pet.define_parameters(LotkaVolterraODE.get_default_fit_params())

    @staticmethod
    def get_default_observables() -> pd.DataFrame:
        return pet.define_observables({"x": "x", "y": "y"}, noise_value=0.02)

    @staticmethod
    def get_default_conditions() -> pd.DataFrame:
        return LotkaVolterraODE.create_conditions([10], [5])


def lotka_volterra_ode() -> sbml.ModelDefinition:

    name = LotkaVolterraODE.name
    print(f"Creating SBML model ({name}).")
    document, model = sbml.create_model(name)

    c = sbml.create_compartment(model, "env", spatialDimensions=0, size=1, units="dimensionless")

    a = sbml.create_parameter(model, "a", value=1.0)
    b = sbml.create_parameter(model, "b", value=1.0)
    c = sbml.create_parameter(model, "c", value=1.0)
    d = sbml.create_parameter(model, "d", value=1.5)

    x = sbml.create_species(model, "x", initialAmount=1.0)
    y = sbml.create_species(model, "y", initialAmount=1.0)

    # dx/dt = a*x - b*x*y
    sbml.create_rate_rule(model, x, formula="a*x - b*x*y")

    # dy/dt = -c*y + d*x*y
    sbml.create_rate_rule(model, y, formula="-c*y + d*x*y")

    return document, model
