from typing import Dict, List

from amici.amici import AmiciSolver  # type: ignore

from polypesto.core import petab as pet
from polypesto.models import ModelBase, sbml

from .common import define_Lowry_I, define_Lowry_I_Temp

########################################################
### Reversible Binary Copolymerization #################
### Lowry Case I: Depropagation from homodyad A only ###
########################################################


class BinaryReversible1Time(ModelBase):

    def _default_obs(self) -> List[str]:
        return ["FA"]

    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
        return {
            "rA": pet.FitParameter(
                id="rA",
                scale=pet.C.LOG10,
                bounds=(1e-3, 1e3),
                nominal_value=1.0,
                estimate=True,
            ),
            "rB": pet.FitParameter(
                id="rB",
                scale=pet.C.LOG10,
                bounds=(1e-3, 1e3),
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
        # return rev_cpe_lowry_caseI()
        return rev_ode_lowry_caseI()

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


def rev_ode_lowry_caseI() -> sbml.ModelDefinition:

    document, model = sbml.init_model("rev_ode_lowry_caseI")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    define_Lowry_I(model, kpAA_constant=True)
    sbml.create_parameter(model, "eps", value=1e-10, units="dimensionless")

    # Define all parameters
    sbml.create_all_parameters(model, ["A", "B", "fA", "fB", "FA", "FB"])
    sbml.create_all_parameters(model, ["fPAA", "fPAB", "fPBA", "fPBB"], value=0.25)
    sbml.create_all_parameters(model, ["A0", "B0"], value=1.0, constant=True)

    # Define all species
    sbml.create_species(model, "R", initialAmount=0.001)
    sbml.create_all_species(
        model, ["RA", "RB", "PAA", "PAB", "PBA", "PBB", "PA", "PB", "xA", "xB"]
    )

    # Define assignment rules
    sbml.create_rule(model, "A", "A0 * (1 - xA)")
    sbml.create_rule(model, "B", "(A0 + B0)*(1 - time) - A")
    sbml.create_rule(model, "fA", "A / (A + B + eps)")
    sbml.create_rule(model, "fB", "1 - fA")
    sbml.create_rule(model, "FA", "(A0/(A0+B0) - (1-x)*fA)/(x+eps)")
    sbml.create_rule(model, "FB", "1-FA")

    sbml.create_rule(model, "PA", "PAA + PBA + RA")
    sbml.create_rule(model, "PB", "PAB + PBB + RB")
    sbml.create_rule(model, "fPAA", "PAA/(PA + eps)")
    sbml.create_rule(model, "fPAB", "PAB/(PB + eps)")
    sbml.create_rule(model, "fPBA", "PBA/(PA + eps)")
    sbml.create_rule(model, "fPBB", "PBB/(PB + eps)")

    rules = {
        "dR_dt": "-R*(kpAA*A + kpBB*B)",
        "dRA_dt": "R*(kpAA*A) - RA*(kpAA*A + kpAB*B)",
        "dRB_dt": "R*(kpBB*B) - RB*(kpBB*B + kpBA*A)",
        "dA_dt": "-A*(kpAA*(R + PA) + kpBA*(R + PB)) + kdAA*PAA",
        "dB_dt": "-B*(kpBB*(R + PB) + kpAB*(R + PA))",
        "dPAA_dt": "kpAA*PA*A - PAA*(kpAA*A + kpAB*B) + kdAA*fPAA*PAA - kdAA*PAA",
        "dPAB_dt": "kpAB*PA*B - PAB*(kpBA*A + kpBB*B)",
        "dPBA_dt": "kpBA*PB*A - PBA*(kpAB*B + kpAA*A) + kdAA*fPBA*PAA",
        "dPBB_dt": "kpBB*PB*B - PBB*(kpBB*B + kpBA*A)",
        "dxA_dt": "-1/A0 * dA_dt",
        "dxB_dt": "-1/B0 * dB_dt",
        "dx_dt": "-1/(A0+B0) * (dA_dt + dB_dt)",
    }
    for var_id, formula in rules.items():
        sbml.create_parameter_rule(model, var_id, formula)

    # Define dxA/dx (dX)
    sbml.create_rate_rule(model, "xA", "dxA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "xB", "dxB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "R", "dR_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "RA", "dRA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "RB", "dRB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PAA", "dPAA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PAB", "dPAB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PBA", "dPBA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PBB", "dPBB_dt/(dx_dt+eps)")

    return sbml.create_model(model, document)


def rev_ode_lowry_caseI_qssa() -> sbml.ModelDefinition:
    """
    Lowry Case I, conversion-domain ODE with QSSA on initiator radicals.

    Drops R, RA, RB (which reach quasi-steady-state effectively
    instantaneously) and pre-loads the four dyad populations at conversion
    zero from the closed-form first-two-step propagation cascade. Reduces
    integrated state count from 9 to 6 (PAA, PAB, PBA, PBB, xA, xB) and
    removes the dominant timescale-separation that drives stiffness in
    rev_ode_lowry_caseI.
    """

    document, model = sbml.init_model("rev_ode_lowry_caseI_qssa")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")

    define_Lowry_I(model, kpAA_constant=True)
    sbml.create_parameter(model, "eps", value=1e-10, units="dimensionless")

    # Algebraic parameters
    sbml.create_all_parameters(model, ["A", "B", "fA", "fB", "FA", "FB"])
    sbml.create_all_parameters(model, ["fPAA", "fPAB", "fPBA", "fPBB"], value=0.25)
    sbml.create_all_parameters(model, ["PA", "PB"], value=0.0)
    sbml.create_all_parameters(model, ["A0", "B0"], value=1.0, constant=True)
    sbml.create_parameter(model, "R0", value=0.001, constant=True)

    # Integrated states: 4 dyads + xA, xB. No R, RA, RB.
    sbml.create_all_species(model, ["PAA", "PAB", "PBA", "PBB", "xA", "xB"])

    # Pre-load dyad populations at conversion 0 from the two-step propagation
    # cascade (R -> RA/RB -> PAA/PAB/PBA/PBB). This bypasses the fast
    # initiation transient entirely.
    sbml.create_initial_assignment(
        model,
        "PAA",
        "R0 * (kpAA*A0/(kpAA*A0 + kpBB*B0)) * (kpAA*A0/(kpAA*A0 + kpAB*B0))",
    )
    sbml.create_initial_assignment(
        model,
        "PAB",
        "R0 * (kpAA*A0/(kpAA*A0 + kpBB*B0)) * (kpAB*B0/(kpAA*A0 + kpAB*B0))",
    )
    sbml.create_initial_assignment(
        model,
        "PBA",
        "R0 * (kpBB*B0/(kpAA*A0 + kpBB*B0)) * (kpBA*A0/(kpBB*B0 + kpBA*A0))",
    )
    sbml.create_initial_assignment(
        model,
        "PBB",
        "R0 * (kpBB*B0/(kpAA*A0 + kpBB*B0)) * (kpBB*B0/(kpBB*B0 + kpBA*A0))",
    )

    # Algebraic rules (note: SBML "time" is total conversion x here)
    sbml.create_rule(model, "A", "A0 * (1 - xA)")
    sbml.create_rule(model, "B", "(A0 + B0)*(1 - time) - A")
    sbml.create_rule(model, "fA", "A / (A + B + eps)")
    sbml.create_rule(model, "fB", "1 - fA")
    sbml.create_rule(model, "FA", "(A0/(A0+B0) - (1-time)*fA)/(time+eps)")
    sbml.create_rule(model, "FB", "1 - FA")

    sbml.create_rule(model, "PA", "PAA + PBA")
    sbml.create_rule(model, "PB", "PAB + PBB")
    sbml.create_rule(model, "fPAA", "PAA/(PA + eps)")
    sbml.create_rule(model, "fPAB", "PAB/(PB + eps)")
    sbml.create_rule(model, "fPBA", "PBA/(PA + eps)")
    sbml.create_rule(model, "fPBB", "PBB/(PB + eps)")

    # Real-time RHS (no R contribution)
    rules = {
        "dA_dt": "-A*(kpAA*PA + kpBA*PB) + kdAA*PAA",
        "dB_dt": "-B*(kpBB*PB + kpAB*PA)",
        "dPAA_dt": "kpAA*PA*A - PAA*(kpAA*A + kpAB*B) + kdAA*fPAA*PAA - kdAA*PAA",
        "dPAB_dt": "kpAB*PA*B - PAB*(kpBA*A + kpBB*B)",
        "dPBA_dt": "kpBA*PB*A - PBA*(kpAB*B + kpAA*A) + kdAA*fPBA*PAA",
        "dPBB_dt": "kpBB*PB*B - PBB*(kpBB*B + kpBA*A)",
        "dxA_dt": "-1/A0 * dA_dt",
        "dxB_dt": "-1/B0 * dB_dt",
        "dx_dt": "-1/(A0+B0) * (dA_dt + dB_dt)",
    }
    for var_id, formula in rules.items():
        sbml.create_parameter_rule(model, var_id, formula)

    # Conversion-domain rate rules: divide real-time RHS by dx/dt
    sbml.create_rate_rule(model, "xA", "dxA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "xB", "dxB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PAA", "dPAA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PAB", "dPAB_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PBA", "dPBA_dt/(dx_dt+eps)")
    sbml.create_rate_rule(model, "PBB", "dPBB_dt/(dx_dt+eps)")

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
    sbml.create_all_species(model, ["A", "B", "xA", "xB", "x"], initialAmount=0.0)
    sbml.create_all_parameters(
        model, ["fA", "fB", "FA", "FB", "dA", "dB", "alpha"], value=0.0
    )

    sbml.create_parameter(model, "A0", value=1.0, constant=True)
    sbml.create_parameter(model, "B0", value=1.0, constant=True)
    sbml.create_parameter(model, "xf", value=1.0, constant=True)

    # Conversion-based bookkeeping
    sbml.create_rule(model, "x", "xf*time")
    sbml.create_rule(model, "A", "A0*(1-xA)")
    sbml.create_rule(model, "B", "(A0+B0)*(1-x)-A")
    sbml.create_rule(model, "xB", "1-B/B0")

    sbml.create_rule(model, "fA", "A/(A+B+eps)")
    sbml.create_rule(model, "fB", "1-fA")
    sbml.create_rule(model, "FA", "(A0/(A0+B0) - (1-x)*fA)/(x+eps)")
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
    sbml.create_rate_rule(model, "xA", "xf * (A0+B0)/A0 * (dA/(dA+dB+eps))")

    return sbml.create_model(model, document)
