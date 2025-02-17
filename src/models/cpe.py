from typing import Optional, List, Sequence, Dict

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import petab.v1.C as C
from src.utils.params import ParameterSet
from src.utils.petab import PetabIO
from src.models.model import Model


# ------------------------------------------------------------------------
# Izu, Wittmer, and Kruger Derivations for CPE
# ------------------------------------------------------------------------
def dfa_dX_izu(X, fA, M0, k):
    """
    Approach: 'izu'
    Returns dfA/dX.
    """
    kpAA, kdAA, kpAB, kdAB, kpBA, kdBA, kpBB, kdBB = k

    # Add small offsets to avoid divisions by zero
    X_val = X + 1e-40
    fA_val = fA + 1e-40

    # A, B
    A = fA_val * (1 - X_val) * M0
    B = (1 - fA_val) * (1 - X_val) * M0

    # Solve "vars" system
    def izu_vars(vars_):
        a, b, c, d = vars_
        eq1 = a + b - 1
        eq2 = a * (kpAB * B + (1 - c) * kdBA) - b * (kpBA * A + (1 - d) * kdAB)
        eq3 = b * c * (1 - d) * kdAB - a * (
            c * (kpAA * A + kdAA + kpAB * B) - (kpAA * A + c * c * kdAA)
        )
        eq4 = a * d * (1 - c) * kdBA - b * (
            d * (kpBB * B + kdBB + kpBA * A) - (kpBB * B + d * d * kdBB)
        )
        return [eq1, eq2, eq3, eq4]

    guess = [0.5, 0.5, 0.5, 0.5]
    res = least_squares(izu_vars, guess, bounds=([0, 0, 0, 0], [1, 1, 1, 1]))
    a, b, c, d = res.x

    dAdt = a * kpAA * A + b * kpBA * A - a * ((1 - c) * kdBA + c * kdAA)
    dBdt = b * kpBB * B + a * kpAB * B - b * ((1 - d) * kdAB + d * kdBB)

    FA = dAdt / (dAdt + dBdt)
    return (fA_val - FA) / (1 - X_val)


def dfa_dX_wittmer(X, fA, M0, k):
    """
    Approach: 'wittmer'
    Returns dfA/dX.
    """
    kpAA, kdAA, kpAB, kdAB, kpBA, kdBA, kpBB, kdBB = k

    X_val = X + 1e-40
    fA_val = fA + 1e-40

    A = fA_val * (1 - X_val) * M0
    B = (1 - fA_val) * (1 - X_val) * M0

    q1 = kdAB / kpBA  # kdAB / kpBA
    q2 = kdBA / kpAB  # kdBA / kpAB
    r1 = kpAA / kpAB  # kpAA / kpAB
    r2 = kpBB / kpBA  # kpBB / kpBA
    kpAA = kdAA / kpAA  # kdAA / kpAA
    kdAA = kdBB / kpBB  # kdBB / kpBB

    def wittmer_vars(vars_):
        x1, y1 = vars_
        eq1 = (
            q1 * y1 * (B + q2 * x1) / (A + q1 * y1)
            - B
            - r1 * (A + kpAA)
            + 2 * r1 * (A + kpAA * (1 - x1) ** 2) / (2 * (1 - x1))
        )
        eq2 = (
            q2 * x1 * (A + q1 * y1) / (B + q2 * x1)
            - A
            - r2 * (B + kdAA)
            + 2 * r2 * (B + kdAA * (1 - y1) ** 2) / (2 * (1 - y1))
        )
        return [eq1, eq2]

    guess = [0.5, 0.5]
    res = least_squares(wittmer_vars, guess, bounds=([0, 0], [1, 1]))
    x1, y1 = res.x

    # Here is the actual ODE expression for dA/dt and dB/dt:
    dAdt = 1 + (r1 * A / B - r1 * kpAA * (1 - x1) / B) / (
        1 - q1 * y1 / B * (B + q2 * x1) / (A + q1 * y1)
    )
    dBdt = 1 + (r2 * B / A - r2 * kdAA * (1 - y1) / A) / (
        1 - q2 * x1 / A * (A + q1 * y1) / (B + q2 * x1)
    )

    FA = dAdt / (dAdt + dBdt)
    return (fA_val - FA) / (1 - X_val)


def dfa_dX_kruger(X, fA, M0, k):
    """
    Approach: 'kruger'
    Returns dfA/dX.
    """
    kpAA, kdAA, kpAB, kdAB, kpBA, kdBA, kpBB, kdBB = k

    X_val = X + 1e-40
    fA_val = fA + 1e-40

    A = fA_val * (1 - X_val) * M0
    B = (1 - fA_val) * (1 - X_val) * M0

    rA = kpAA / kpAB  # kAA / kAB
    rB = kpBB / kpBA  # kBB / kBA
    RA = kdAB / kpBA  # kdAB / kpBA
    RAA = kdAA / kpAB  # kdAA / kpAB
    RB = kdBA / kpAB  # kdBA / kpAB
    RBB = kdBB / kpBA  # kdBB / kpBA

    def kruger_vars(vars_):
        PAA, PAB, PBA, PBB = vars_
        a = 1 - RA * PBA / (A + RA * PBA)
        b = RAA - RA * RB * PBA / (A + RA * PBA)
        c = 1 - RB * PAB / (B + RB * PAB)
        d = RBB - RB * RA * PAB / (B + RB * PAB)

        eq1 = b * PAB * PAB + (rA * A + B * a - b) * PAB - B * a
        eq2 = d * PBA * PBA + (rB * B + A * c - d) * PBA - A * c
        eq3 = PAA + PAB - 1
        eq4 = PBB + PBA - 1
        return [eq1, eq2, eq3, eq4]

    guess = [0.5, 0.5, 0.5, 0.5]
    res = least_squares(kruger_vars, guess, bounds=([0, 0, 0, 0], [1, 1, 1, 1]))
    PAA, PAB, PBA, PBB = res.x

    dAdt = A * (rA * (A + RA * PBA) + B - RAA * PAA) - RA * PBA * (RAA * PAA + RB * PAB)
    dBdt = B * (rB * (B + RB * PAB) + A - RBB * PBB) - RB * PAB * (RBB * PBB + RA * PBA)

    FA = dAdt / (dAdt + dBdt)
    return (fA_val - FA) / (1 - X_val)


# ------------------------------------------------------------------------
# Copolymer Equation Model
# ------------------------------------------------------------------------
class CPE:
    """
    Reaction rate constants are passed like so:
        inputs = [kpAA, kdAA, kpAB, kdAB, kpBA, kdBA, kpBB, kdBB]

    Example usage:
        model = CPE([1.0, 0.5, 0.8, 0.6, 0.3, 0.2, 0.7, 0.1])
        X, xA, xB = model.solve(fA0=0.4, M0=1.0, approach='izu')
    """

    def __init__(self, inputs: List[float]):
        # Reaction rate constants
        self.k = inputs

        # We define a dictionary to map approach names -> the outside functions.
        self.approach_funcs = {
            "izu": dfa_dX_izu,
            "wittmer": dfa_dX_wittmer,
            "kruger": dfa_dX_kruger,
        }

    @staticmethod
    def from_param_set(k: ParameterSet):
        """
        inputs = [rA, rB, rX, KAA, KAB, KBA, KBB]
        """
        rA = k.by_id("rA").value
        rB = k.by_id("rB").value
        rX = k.by_id("rX").value
        KAA = k.by_id("KAA").value
        KAB = k.by_id("KAB").value
        KBA = k.by_id("KBA").value
        KBB = k.by_id("KBB").value

        kpAA = 1.0
        kpAB = kpAA / rA
        kpBB = kpAA / rX
        kpBA = kpBB / rB

        kdAA = KAA * kpAA
        kdAB = KAB * kpAB
        kdBA = KBA * kpBA
        kdBB = KBB * kpBB

        return CPE([kpAA, kdAA, kpAB, kdAB, kpBA, kdBA, kpBB, kdBB])

    @staticmethod
    def from_ratios(inputs: List[float]) -> "CPE":
        """
        inputs = [rA, rB, rX, KAA, KAB, KBA, KBB]
        """
        rA, rB, rX, KAA, KAB, KBA, KBB = inputs

        kpAA = 1.0
        kpAB = kpAA / rA
        kpBB = kpAA / rX
        kpBA = kpBB / rB

        kdAA = KAA * kpAA
        kdAB = KAB * kpAB
        kdBA = KBA * kpBA
        kdBB = KBB * kpBB

        return CPE([kpAA, kdAA, kpAB, kdAB, kpBA, kdBA, kpBB, kdBB])

    def solve(
        self,
        fA0: float,
        M0: float,
        approach: str = "izu",
        t_eval: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Solve the ODE system for fA vs X using the chosen approach.
        """
        if approach not in self.approach_funcs:
            raise ValueError(f"Unknown approach '{approach}'")

        # Setup the time points
        if t_eval is None:
            t_eval = np.linspace(0, 1, 100)
        bounds = (t_eval[0], t_eval[-1])

        # Setup the ODE function
        ode_func = self.approach_funcs[approach]

        # Wrap in a function that solve_ivp expects: f(X, [fA]) -> [dfA/dX]
        def wrapper(X, y):
            return [ode_func(X, y[0], M0, self.k)]

        # Solve via solve_ivp
        # print(fA0)
        sol = solve_ivp(fun=wrapper, t_span=bounds, y0=[fA0], t_eval=t_eval, **kwargs)

        X_sol = sol.t
        fA_sol = sol.y[0]

        # Post-processing
        # A(X) = fA(X) * (1 - X) * M0
        # B(X) = (1 - fA(X)) * (1 - X) * M0
        A_vals = fA_sol * (1 - X_sol) * M0
        B_vals = (1 - fA_sol) * (1 - X_sol) * M0

        A0 = fA0 * M0  # initial A
        B0 = (1 - fA0) * M0  # initial B

        # Conversions
        xA = (A0 - A_vals) / (A0 + 1e-40)
        xB = (B0 - B_vals) / (B0 + 1e-40)

        return X_sol, xA, xB


def run_CPE_sim(
    model: CPE,
    t_eval: Sequence[float],
    conditions: Dict[str, float],
    sigma: float = 0.0,
    approach: str = "izu",
    **kwargs,
) -> Dict[str, np.ndarray]:

    fA0 = 0.0
    M0 = 0.0

    if "fA0" in conditions and "M0" in conditions:
        fA0 = conditions["fA0"]
        M0 = conditions["M0"]
    elif "A0" in conditions and "B0" in conditions:
        M0 = conditions["A0"] + conditions["B0"]
        fA0 = conditions["A0"] / M0
    else:
        raise Exception("Invalid input for conditions.")

    # print(fA0)
    # print(M0)
    # print(type(fA0))
    # print(type(M0))

    X_sol, xA, xB = model.solve(
        fA0=float(fA0), M0=float(M0), approach=approach, t_eval=t_eval, **kwargs
    )

    cpe_outputs = {"X_sol": X_sol, "xA": xA, "xB": xB}
    return cpe_outputs


def get_meas_from_cpe_sim(
    cpe_output: Dict[str, np.ndarray],
    observables_df: pd.DataFrame,
    cond_id: str = "none",
    obs_sigma: float = 0.00,
) -> pd.DataFrame:
    # Transform CPE_output to measurements_df

    # This should throw an error if observables_df has anything other
    # than 'xA' and 'xB' in the C.FORMULA column

    meas_dfs = []
    for obs_id, row in observables_df.iterrows():
        observables = row.to_dict()
        # print(observables)

        obs_formula = observables[C.OBSERVABLE_FORMULA]
        if obs_formula not in ["xA", "xB"]:
            raise Exception("Invalid observable.")

        # print(cpe_output)
        obs_data = cpe_output[obs_formula]
        obs_data = np.array(obs_data) * (1 + obs_sigma * np.random.randn(len(obs_data)))
        num_pts = len(obs_data)

        obs_meas_df = pd.DataFrame(
            {
                C.OBSERVABLE_ID: [obs_id] * num_pts,
                C.SIMULATION_CONDITION_ID: [cond_id] * num_pts,
                C.TIME: cpe_output["X_sol"],
                C.MEASUREMENT: obs_data,
            }
        )
        meas_dfs.append(obs_meas_df)
    meas_df = pd.concat(meas_dfs, ignore_index=True)

    meas_df = PetabIO.format_meas_df(meas_df)

    return meas_df


def define_measurements_cpe(
    cpe_model: CPE,
    t_eval: Sequence[float],
    conditions_df: pd.DataFrame,
    observables_df: pd.DataFrame,
    obs_sigma: float = 0.00,
    meas_sigma: float = 0.005,
    approach: str = "izu",
    **kwargs,
) -> pd.DataFrame:

    measurement_dfs = []

    for cond_id, row in conditions_df.iterrows():
        # Extract conditions for this row as a dictionary
        conditions = row.to_dict()

        # Run the simulation with these conditions
        cpe_output = run_CPE_sim(
            cpe_model, t_eval, conditions, sigma=meas_sigma, **kwargs
        )

        # Generate measurements from the simulation
        meas_df = get_meas_from_cpe_sim(
            cpe_output, observables_df, cond_id=str(cond_id), obs_sigma=obs_sigma
        )
        measurement_dfs.append(meas_df)

    measurement_df = pd.concat(measurement_dfs, ignore_index=True)

    return measurement_df


class CPEModel(Model):

    def __init__(
        self,
        name: str,
        model: CPE,
        obs_df: pd.DataFrame,
    ):
        super().__init__(name, model, obs_df)

    def simulate(
        self,
        t_eval: List[float],
        conditions: Dict[str, float],
        cond_id: str = None,
        approach: str = "izu",
        **kwargs,
    ) -> pd.DataFrame:
       
       cpe_data = run_CPE_sim(self.model, t_eval, conditions, approach=approach, **kwargs)
       return get_meas_from_cpe_sim(cpe_data, self.obs_df, cond_id=cond_id)

    def set_params(self, param_set: ParameterSet):
        self.model = CPE.from_param_set(param_set)


def create_model(
    name: str,
    obs_df: pd.DataFrame,
) -> CPEModel:
    
    return CPEModel(name, None, obs_df)
