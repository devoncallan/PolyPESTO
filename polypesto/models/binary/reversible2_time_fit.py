from typing import Dict, List

from amici.amici import AmiciSolver  # type: ignore

from polypesto.core import petab as pet
from polypesto.models import ModelBase, sbml

from .common import define_Lowry_II_Temp_Fit

###################################################################
### Reversible Binary Copolymerization (time-parameterised) #######
### Lowry Case II: depropagation from BOTH AA and BA dyads ########
### KAA(T) = KAA_Tref * exp(-dH_R*(1/T_K - 1/Tref))
### KBA    = f_BA * KAA  (f_BA fittable; f_BA=0 reduces to Case I)
###################################################################

# Same Tref convention as BinaryReversible1TimeFit so KAA_Tref / dH_R priors
# carry over without re-calibration.
TREF_K = 350.0


class BinaryReversible2TimeFit(ModelBase):

    def _default_obs(self) -> List[str]:
        return ["FA"]

    def _default_fit_params(self) -> Dict[str, pet.FitParameter]:
        # KAA_Tref / dH_R priors copied verbatim from BinaryReversible1TimeFit
        # (Tsarevsky NLS, 4 x SE inflation). f_BA defaults to estimate=False
        # so the model collapses to Case I unless explicitly turned on.
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
            "KAA_Tref": pet.FitParameter(
                id="KAA_Tref",
                scale=pet.C.LIN,
                bounds=(0.5, 5.0),
                nominal_value=2.27,
                estimate=True,
                prior_type="normal",
                prior_params="2.27;0.144",  # 4 x SE from Tsarevsky NLS
            ),
            "dH_R": pet.FitParameter(
                id="dH_R",
                scale=pet.C.LIN,
                bounds=(500.0, 4000.0),
                nominal_value=1893.0,
                estimate=True,
                prior_type="normal",
                prior_params="1893.0;400.0",  # 4 x SE from Tsarevsky NLS
            ),
            "f_BA": pet.FitParameter(
                id="f_BA",
                scale=pet.C.LIN,
                bounds=(0.0, 1.0),
                nominal_value=0.0,
                estimate=False,
            ),
        }

    def _default_sbml_model(self) -> sbml.ModelDefinition:
        return rev_cpe_lowry_caseII_time_fit()

    def _default_solver_options(self, solver: AmiciSolver) -> AmiciSolver:
        solver.setNewtonMaxSteps(1_000)
        solver.setNewtonDampingFactorMode(1)
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


def rev_cpe_lowry_caseII_time_fit() -> sbml.ModelDefinition:
    """
    Conversion-as-time SBML model with Lowry Case II depropagation:
    both AA and BA dyads can depropagate (B side fully irreversible).

    The dyad balance for alpha = pAA is identical to Case I because kdBA
    depropagation removes mass from the A-terminal pool entirely (BA -> *B)
    rather than redistributing within the AA/BA dyad split. The Case II
    corrections enter through pA (extra A->B drain) and dA (extra A release).
    """

    document, model = sbml.init_model("rev_cpe_lowry_caseII_time_fit")
    sbml.create_compartment(model, "c", spatialDimensions=0, units="dimensionless")
    sbml.create_parameter(model, "eps", value=1e-10)

    # Per-condition initial / experimental parameters
    sbml.create_parameter(model, "T_K", value=313.15, constant=True)
    sbml.create_parameter(model, "A0", value=1.0, constant=True)
    sbml.create_parameter(model, "B0", value=1.0, constant=True)
    sbml.create_parameter(model, "xf", value=1.0, constant=True)

    # Rate parameters: creates kpAA, kpAB, kpBA, kpBB, kdAA, kdBA, KAA, KBA,
    # KAA_Tref, dH_R, f_BA, Tref with Van't Hoff KAA(T) and KBA = f_BA*KAA.
    define_Lowry_II_Temp_Fit(model, Tref=TREF_K, kpAA_constant=True)

    # Species / state
    sbml.create_all_species(model, ["A", "B", "xA", "xB", "x"], initialAmount=0.0)
    sbml.create_all_parameters(
        model, ["fA", "fB", "FA", "FB", "dA", "dB", "alpha"], value=0.0
    )

    # Conversion-based bookkeeping (time = scaled conversion clock; x = xf*time)
    sbml.create_rule(model, "x", "xf*time")
    sbml.create_rule(model, "A", "A0*(1-xA)")
    sbml.create_rule(model, "B", "(A0+B0)*(1-x)-A")
    sbml.create_rule(model, "xB", "1-B/B0")

    sbml.create_rule(model, "fA", "A/(A+B+eps)")
    sbml.create_rule(model, "fB", "1-fA")
    sbml.create_rule(model, "FA", "(A0/(A0+B0) - (1-x)*fA)/(x+eps)")
    sbml.create_rule(model, "FB", "1-FA")

    # Lowry Case II alpha = pAA from stable quadratic root (same quadratic as
    # Case I: dyad balance only sees AA<->BA propagation flux and kdAA's
    # self-resampling; kdBA exits the A-terminal pool entirely).
    sbml.create_parameter(model, "u", value=0.0)
    sbml.create_parameter(model, "S", value=0.0)
    sbml.create_parameter(model, "D", value=0.0)

    sbml.create_rule(model, "u", "kpAA*A")
    sbml.create_rule(model, "S", "kpAA*A + kpAB*B + kdAA")
    sbml.create_rule(model, "D", "piecewise(S*S - 4*kdAA*u, S*S - 4*kdAA*u >= 0, eps)")

    sbml.create_rule(
        model,
        "alpha",
        "piecewise( u/(u + kpAB*B + eps), kdAA < eps, (2*u)/(S + sqrt(D)) )",
    )

    # Terminal chain-end fraction. Case II adds kdBA*(1-alpha) to the A->B
    # drain (BA depropagation kicks the chain to terminal B).
    sbml.create_parameter(model, "pA", value=0.5)
    sbml.create_parameter(model, "pB", value=0.5)
    sbml.create_rule(
        model, "pA", "(kpBA*A)/(kpBA*A + kpAB*B + kdBA*(1 - alpha) + eps)"
    )
    sbml.create_rule(model, "pB", "1 - pA")

    # Normalised monomer rates. Case II adds kdBA*(1-alpha) regeneration of A.
    sbml.create_rule(
        model,
        "dA",
        "-A*(kpAA*pA + kpBA*pB) + pA*(kdAA*alpha + kdBA*(1 - alpha))",
    )
    sbml.create_rule(model, "dB", "-B*(kpBB*pB + kpAB*pA)")

    # dxA/dx_total (SBML time is the conversion clock; xf scales it)
    sbml.create_rate_rule(model, "xA", "xf * (A0+B0)/A0 * (dA/(dA+dB+eps))")

    return sbml.create_model(model, document)
