from typing import Dict, List

from polypesto.core import petab as pet
from polypesto.models import sbml, ModelBase


################################
### Lotka-Volterra Equations ###
################################

# dx/dt = a*x - b*x*y
# dy/dt = -c*y + d*x*y


class LotkaVolterra(ModelBase):
    """Lotka Volterra Model"""

    def _default_obs(self) -> List[str]:
        return ["x", "y"]

    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
        return {
            "a": pet.FitParameter(
                id="a",
                scale=pet.C.LOG10,
                bounds=(0.001, 2),
                nominal_value=1.0,
                estimate=True,
            ),
            "b": pet.FitParameter(
                id="b",
                scale=pet.C.LOG10,
                bounds=(0.001, 1),
                nominal_value=0.1,
                estimate=True,
            ),
            "c": pet.FitParameter(
                id="c",
                scale=pet.C.LOG10,
                bounds=(0.001, 2),
                nominal_value=0.075,
                estimate=True,
            ),
            "d": pet.FitParameter(
                id="d",
                scale=pet.C.LOG10,
                bounds=(0.001, 1),
                nominal_value=1.5,
                estimate=True,
            ),
        }

    def _default_sbml_model(self) -> sbml.ModelDefinition:
        return lotka_volterra_ode()


def lotka_volterra_ode() -> sbml.ModelDefinition:

    document, model = sbml.init_model("lotka_volterra_ode")

    sbml.create_compartment(
        model, "env", spatialDimensions=0, size=1, units="dimensionless"
    )

    sbml.create_parameter(model, "a", value=1.0)
    sbml.create_parameter(model, "b", value=1.0)
    sbml.create_parameter(model, "c", value=1.0)
    sbml.create_parameter(model, "d", value=1.5)

    sbml.create_species(model, "x", initialAmount=1.0)
    sbml.create_species(model, "y", initialAmount=1.0)

    # dx/dt = a*x - b*x*y
    sbml.create_rate_rule(model, "x", formula="a*x - b*x*y")

    # dy/dt = -c*y + d*x*y
    sbml.create_rate_rule(model, "y", formula="-c*y + d*x*y")

    return sbml.create_model(model, document)
