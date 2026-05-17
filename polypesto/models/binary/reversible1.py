from typing import Dict, List

from amici.amici import AmiciSolver  # type: ignore

from polypesto.core import petab as pet
from polypesto.models import ModelBase, sbml

from .common import define_Lowry_I, define_Lowry_I_Temp

########################################################
### Reversible Binary Copolymerization #################
### Lowry Case I: Depropagation from homodyad A only ###
########################################################


class BinaryReversible1(ModelBase):

    def _default_obs(self) -> List[str]:
        return ["FA"]

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
            "KAA": pet.FitParameter(
                id="KAA",
                scale=pet.C.LOG10,
                bounds=(1e-2, 1e2),
                nominal_value=1.0,
                estimate=True,
            ), 
        }

    def _default_sbml_model(self) -> sbml.ModelDefinition:
        return rev_cpe_lowry_caseI()

    def _default_solver_options(self, solver: AmiciSolver) -> AmiciSolver:
        solver.setNewtonMaxSteps(1_000)
        solver.setNewtonDampingFactorMode(1)
        # solver.setAbsoluteTolerance(1e-10)
        solver.setAbsoluteTolerance(1e-8)
        solver.setRelativeTolerance(1e-6)
        solver.setMaxSteps(10_000)
        solver.setMaxConvFails(1_000)
        solver.setMaxNonlinIters(10)
        solver.setLinearSolver(9)
        solver.setStabilityLimitFlag(True)
        solver.setReturnDataReportingMode(0)
        solver.setLinearMultistepMethod(2)
        return solver


def rev_cpe_lowry_caseI_temp() -> sbml.ModelDefinition:
    """
    Lowry Case I: only A depropagates, only from AA.
    No explicit dyad variables; use closed-form pA and alpha.
    """

    document, model = sbml.init_model("rev_cpe_lowry_caseI")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")
    sbml.create_parameter(model, "eps", value=1e-10)

    # Initial conditions
    sbml.create_parameter(model, "T_K", value=298.15, constant=True)
    sbml.create_parameter(model, "A0", value=1.0, constant=True)
    sbml.create_parameter(model, "B0", value=1.0, constant=True)

    # Define rate parameters (must create kpAA,kpAB,kpBA,kpBB and kdAA,...)
    define_Lowry_I_Temp(model, kpAA_constant=True)

    # Species / state
    sbml.create_all_species(model, ["A", "B", "xA", "xB"], initialAmount=0.0)
    sbml.create_all_parameters(
        model, ["fA", "fB", "FA", "FB", "dA", "dB", "alpha"], value=0.0
    )

    # Conversion-based bookkeeping (same pattern you used)
    sbml.create_rule(model, "A", "A0*(1-xA)")
    sbml.create_rule(model, "B", "(A0+B0)*(1-time)-A")
    sbml.create_rule(model, "xB", "1-B/B0")

    sbml.create_rule(model, "fA", "A/(A+B+eps)")
    sbml.create_rule(model, "fB", "1-fA")
    sbml.create_rule(model, "FA", "(A0/(A0+B0) - (1-time)*fA)/(time+eps)")
    sbml.create_rule(model, "FB", "1-FA")

    # Terminal chain-end fraction (no algebraic solve needed)
    # pA and pB are "probabilities" -> easiest as parameters via rules
    sbml.create_parameter(model, "pA", value=0.5)
    sbml.create_parameter(model, "pB", value=0.5)
    sbml.create_rule(model, "pA", "(kpBA*A)/(kpBA*A + kpAB*B + eps)")
    sbml.create_rule(model, "pB", "1 - pA")

    # Lowry Case I alpha = pAA from stable quadratic root
    # u = kpAA*A
    # S = kpAA*A + kpAB*B + kdAA
    # D = S^2 - 4*kdAA*u
    sbml.create_parameter(model, "u", value=0.0)
    sbml.create_parameter(model, "S", value=0.0)
    sbml.create_parameter(model, "D", value=0.0)

    sbml.create_rule(model, "u", "kpAA*A")
    sbml.create_rule(model, "S", "kpAA*A + kpAB*B + kdAA")
    sbml.create_rule(model, "D", "piecewise(S*S - 4*kdAA*u, S*S - 4*kdAA*u >= 0, eps)")

    # If you want extra robustness when kdAA ~ 0, use a piecewise fallback.
    sbml.create_rule(
        model,
        "alpha",
        "piecewise( u/(u + kpAB*B + eps), kdAA < eps, (2*u)/(S + sqrt(D)) )",
    )

    # Normalized monomer rates (your same normalization logic)
    sbml.create_rule(model, "dA", "-A*(kpAA*pA + kpBA*pB) + pA*(kdAA*alpha)")
    sbml.create_rule(model, "dB", "-B*(kpBB*pB + kpAB*pA)")

    # dxA/dx_total (SBML time is your x_total)
    sbml.create_rate_rule(model, "xA", "(A0+B0)/A0 * (dA/(dA+dB+eps))")

    return sbml.create_model(model, document)


def rev_cpe_lowry_caseI() -> sbml.ModelDefinition:
    """
    Lowry Case I: only A depropagates, only from AA.
    No explicit dyad variables; use closed-form pA and alpha.
    """

    document, model = sbml.init_model("rev_cpe_lowry_caseI")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")
    sbml.create_parameter(model, "eps", value=1e-10)

    # Define rate parameters (must create kpAA,kpAB,kpBA,kpBB and kdAA,...)
    define_Lowry_I(model, kpAA_constant=True)

    # Species / state
    sbml.create_all_species(model, ["A", "B", "xA", "xB"], initialAmount=0.0)
    sbml.create_all_parameters(
        model, ["fA", "fB", "FA", "FB", "dA", "dB", "alpha"], value=0.0
    )

    sbml.create_parameter(model, "A0", value=1.0, constant=True)
    sbml.create_parameter(model, "B0", value=1.0, constant=True)

    # Conversion-based bookkeeping (same pattern you used)
    sbml.create_rule(model, "A", "A0*(1-xA)")
    sbml.create_rule(model, "B", "(A0+B0)*(1-time)-A")
    sbml.create_rule(model, "xB", "1-B/B0")

    sbml.create_rule(model, "fA", "A/(A+B+eps)")
    sbml.create_rule(model, "fB", "1-fA")
    sbml.create_rule(model, "FA", "(A0/(A0+B0) - (1-time)*fA)/(time+eps)")
    sbml.create_rule(model, "FB", "1-FA")

    # Terminal chain-end fraction (no algebraic solve needed)
    # pA and pB are "probabilities" -> easiest as parameters via rules
    sbml.create_parameter(model, "pA", value=0.5)
    sbml.create_parameter(model, "pB", value=0.5)
    sbml.create_rule(model, "pA", "(kpBA*A)/(kpBA*A + kpAB*B + eps)")
    sbml.create_rule(model, "pB", "1 - pA")

    # Lowry Case I alpha = pAA from stable quadratic root
    # u = kpAA*A
    # S = kpAA*A + kpAB*B + kdAA
    # D = S^2 - 4*kdAA*u
    sbml.create_parameter(model, "u", value=0.0)
    sbml.create_parameter(model, "S", value=0.0)
    sbml.create_parameter(model, "D", value=0.0)

    sbml.create_rule(model, "u", "kpAA*A")
    sbml.create_rule(model, "S", "kpAA*A + kpAB*B + kdAA")
    sbml.create_rule(model, "D", "piecewise(S*S - 4*kdAA*u, S*S - 4*kdAA*u >= 0, eps)")

    # If you want extra robustness when kdAA ~ 0, use a piecewise fallback.
    sbml.create_rule(
        model,
        "alpha",
        "piecewise( u/(u + kpAB*B + eps), kdAA < eps, (2*u)/(S + sqrt(D)) )",
    )

    # Normalized monomer rates (your same normalization logic)
    sbml.create_rule(model, "dA", "-A*(kpAA*pA + kpBA*pB) + pA*(kdAA*alpha)")
    sbml.create_rule(model, "dB", "-B*(kpBB*pB + kpAB*pA)")

    # dxA/dx_total (SBML time is your x_total)
    sbml.create_rate_rule(model, "xA", "(A0+B0)/A0 * (dA/(dA+dB+eps))")

    return sbml.create_model(model, document)
