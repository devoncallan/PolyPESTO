from typing import Dict, List

from amici import AmiciSolver, Solver  # type: ignore

from polypesto.core import petab as pet
from polypesto.models import ModelBase, sbml

from .common import define_irreversible_k

############################################
### Irreversible Binary Copolymerization ###
############################################


class BinaryIrreversible(ModelBase):

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
        return irr_cpe()

    def _default_solver_options(self, solver: AmiciSolver) -> AmiciSolver:
        
        # print("SOLVER OPTIONS in _default_solver_options (before):")
        # print(solver.getSensitivityMethod())
        # print(solver.getSensitivityOrder())
        # print(solver.getReturnDataReportingMode())
        # print("END SOLVER OPTIONS")
        
        solver.setSensitivityMethod(1)
        solver.setSensitivityOrder(1)
        solver.setReturnDataReportingMode(0)
        solver.setRelativeTolerance(1e-6)
        solver.setAbsoluteTolerance(1e-8)
        
        # print("SOLVER OPTIONS in _default_solver_options:")
        # print(solver.getSensitivityMethod())
        # print(solver.getSensitivityOrder())
        # print(solver.getReturnDataReportingMode())
        # print("END SOLVER OPTIONS")

        return solver


def irr_cpe() -> sbml.ModelDefinition:

    document, model = sbml.init_model("irr_cpe")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")
    sbml.create_parameter(model, "eps", value=1e-10)

    # Define reaction rate parameters
    define_irreversible_k(model)

    # Initialize all species and parameters
    sbml.create_all_species(model, ["A", "B", "xA", "xB"], initialAmount=0.0)
    sbml.create_all_parameters(model, ["fA", "fB", "FA", "FB", "dA", "dB"])

    # Define initial species
    sbml.create_parameter(model, "A0", value=1.0, constant=True)
    sbml.create_parameter(model, "B0", value=1.0, constant=True)

    # Define species and parameters
    sbml.create_rule(model, "A", "A0*(1-xA)")
    sbml.create_rule(model, "B", "(A0+B0)*(1-time)-A")
    sbml.create_rule(model, "xB", "1-B/B0")

    sbml.create_rule(model, "fA", "A/(A+B+eps)")
    sbml.create_rule(model, "fB", "1-fA")
    sbml.create_rule(model, "FA", "(A0/(A0+B0) - (1-time)*fA)/(time+eps)")
    sbml.create_rule(model, "FB", "1-FA")

    sbml.create_rule(model, "dA", "-A*(rA*A+B)")
    sbml.create_rule(model, "dB", "-B*(rB*B+A)")

    # Define differential equation
    sbml.create_rate_rule(model, "xA", "(A0+B0)/A0 * ((dA+eps)/(dA+dB+eps))")

    return sbml.create_model(model, document)


def irr_ode() -> sbml.ModelDefinition:

    document, model = sbml.init_model("irr_ode")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")
    sbml.create_parameter(model, "eps", value=1e-10)

    # Define reaction rate parameters
    define_irreversible_k(model, kpAA_constant=True)

    # Initialize all species and parameters
    sbml.create_all_species(model, ["PA", "PB", "xA", "xB"])
    sbml.create_all_parameters(model, ["A", "B", "fA", "fB", "FA", "FB"])
    sbml.create_all_parameters(
        model,
        ["dR_dt", "dA_dt", "dB_dt", "dPA_dt", "dPB_dt", "dxA_dt", "dxB_dt", "dx_dt"],
    )

    # Define initial species
    sbml.create_species(model, "R", initialAmount=0.001)
    sbml.create_parameter(model, "A0", value=1.0, constant=True)
    sbml.create_parameter(model, "B0", value=1.0, constant=True)

    # Define species and parameters
    sbml.create_rule(model, "A", "A0*(1-xA)")
    sbml.create_rule(model, "B", "(A0+B0)*(1-time)-A")
    sbml.create_rule(model, "fA", "A/(A+B+eps)")
    sbml.create_rule(model, "fB", "1-fA")
    sbml.create_rule(model, "FA", "(A0/(A0+B0) - (1-time)*fA)/(time+eps)")
    sbml.create_rule(model, "FB", "1-FA")

    # Define rates of change
    sbml.create_rule(model, "dR_dt", "-R*(kpAA*A + kpBB*B)")
    sbml.create_rule(model, "dA_dt", "-A*(kpAA*(R+PA) + kpBA*(R+PB))")
    sbml.create_rule(model, "dB_dt", "-B*(kpBB*(R+PB) + kpAB*(R+PA))")
    sbml.create_rule(model, "dPA_dt", "A*(kpAA*(R+PA)+kpBA*PB) - PA*(kpAA*A+kpAB*B)")
    sbml.create_rule(model, "dPB_dt", "B*(kpBB*(R+PB)+kpAB*PA) - PB*(kpBB*B + kpBA*A)")

    sbml.create_rule(model, "dxA_dt", "-1/A0 * dA_dt")
    sbml.create_rule(model, "dxB_dt", "-1/B0 * dB_dt")
    sbml.create_rule(model, "dx_dt", "-1/(A0+B0) * (dA_dt+dB_dt)")

    # Define differential equations
    sbml.create_rate_rule(model, "R", "dR_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PA", "dPA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PB", "dPB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "xA", "dxA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "xB", "dxB_dt/(dx_dt+eps)")

    return sbml.create_model(model, document)
