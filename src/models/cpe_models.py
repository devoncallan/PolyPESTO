import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.integrate import odeint, solve_ivp


class CopolymerEquations:

    def __init__(self, inputs):

        self.fA0 = inputs[0]
        self.fB0 = 1 - self.fA0
        self.M0 = inputs[9]

        self.kpAA = self.k1 = inputs[1]
        self.kdAA = self.k2 = inputs[2]
        self.kpAB = self.k3 = inputs[3]
        self.kdAB = self.k4 = inputs[4]
        self.kpBA = self.k5 = inputs[5]
        self.kdBA = self.k6 = inputs[6]
        self.kpBB = self.k7 = inputs[7]
        self.kdBB = self.k8 = inputs[8]

        self.bounds = [0, 1.0]
        self.num_points = 100
        self.t_eval = np.linspace(self.bounds[0], self.bounds[1], self.num_points)
        self.method = "BDF"

    def set_bounds(self, bounds):
        self.bounds = bounds
        self.t_eval = np.linspace(self.bounds[0], self.bounds[1], self.num_points)

    @staticmethod
    def fromLundberg(inputs, M0=1.0):

        f1, r1, r2, b1, g1, b2, g2 = inputs
        new_inputs = [f1, 1, b1, 1 / r1, g2, 1 / r2, g1, 1, b2, M0]

        return CopolymerEquations(new_inputs)

    @staticmethod
    def from_ratios(inputs, fA0=0.5, M0=1.0):
        """
        rA, rB, rX, KAA, KAB, KBA, KBB = inputs
        """
        rA, rB, rX, KAA, KAB, KBA, KBB = inputs
        kpAA = 1  # rA = kpAA/kpAB
        kpAB = kpAA / rA
        kpBB = kpAA / rX
        kpBA = kpBB / rB
        kdAA = KAA * kpAA
        kdAB = KAB * kpAB
        kdBA = KBA * kpBA
        kdBB = KBB * kpBB
        new_inputs = [fA0, kpAA, kdAA, kpAB, kdAB, kpBA, kdBA, kpBB, kdBB, M0]

        return CopolymerEquations(new_inputs)

    def solve_izu(self):

        def izu_vars(X, A, B):
            def izu_var_eqns(vars):
                a, b, c, d = vars
                eq1 = a + b - 1
                eq2 = a * (self.k3 * B + (1 - c) * self.k6) - b * (
                    self.k5 * A + (1 - d) * self.k4
                )
                eq3 = b * c * (1 - d) * self.k4 - a * (
                    c * (self.k1 * A + self.k2 + self.k3 * B)
                    - (self.k1 * A + c * c * self.k2)
                )
                eq4 = a * d * (1 - c) * self.k6 - b * (
                    d * (self.k7 * B + self.k8 + self.k5 * A)
                    - (self.k7 * B + d * d * self.k8)
                )
                return [eq1, eq2, eq3, eq4]

            initial_guess = [0.5, 0.5, 0.5, 0.5]
            res = least_squares(
                izu_var_eqns, initial_guess, bounds=([0, 0, 0, 0], [1, 1, 1, 1])
            )
            # print(X, res.cost, res.x)
            return res.x
            # initial_guess = [0.5, 0.5, 0.5, 0.5]
            # return fsolve(witmer_var_eqns, initial_guess,xtol=1e-10, maxfev=1000)

        def izu_eqns(X, fA):
            fA = fA[0]
            A = fA * (1 - X) * (self.fA0 + self.fB0) * self.M0
            B = (1 - fA) * (1 - X) * (self.fA0 + self.fB0) * self.M0
            a, b, c, d = izu_vars(X, A, B)

            dAdt = (
                a * self.k1 * A
                + b * self.k5 * A
                - a * ((1 - c) * self.k6 + c * self.k2)
            )
            dBdt = (
                b * self.k7 * B
                + a * self.k3 * B
                - b * ((1 - d) * self.k4 + d * self.k8)
            )
            FA = dAdt / (dAdt + dBdt)
            dfA_dX = (fA - FA) / (1 - X)

            return [dfA_dX]

        sol = solve_ivp(
            izu_eqns,
            self.bounds,
            [self.fA0],
            method=self.method,
            max_step=0.001,
            t_eval=self.t_eval,
        )
        X = sol.t
        fA = sol.y[0]

        A = fA * (1 - X) * (self.fA0 + self.fB0)
        B = (1 - fA) * (1 - X) * (self.fA0 + self.fB0)

        xA = (self.fA0 - A) / (self.fA0)
        xB = (self.fB0 - B) / (self.fB0)

        return X, xA, xB

    def solve_wittmer(self):

        q1 = self.kdAB / self.kpBA
        q2 = self.kdBA / self.kpAB
        r1 = self.kpAA / self.kpAB
        r2 = self.kpBB / self.kpBA
        K1 = self.kdAA / self.kpAA
        K2 = self.kdBB / self.kpBB

        def wittmer_vars(X, A, B):
            def wittmer_var_eqns(vars):
                x1, y1 = vars
                eq1 = (
                    q1 * y1 * (B + q2 * x1) / (A + q1 * y1)
                    - B
                    - r1 * (A + K1)
                    + 2 * r1 * (A + K1 * (1 - x1) ** 2) / (2 * (1 - x1))
                )
                eq2 = (
                    q2 * x1 * (A + q1 * y1) / (B + q2 * x1)
                    - A
                    - r2 * (B + K2)
                    + 2 * r2 * (B + K2 * (1 - y1) ** 2) / (2 * (1 - y1))
                )
                return [eq1, eq2]

            initial_guess = [0.5, 0.5]
            # return fsolve(wittmer_var_eqns, initial_guess,xtol=1e-10, maxfev=1000)
            res = least_squares(
                wittmer_var_eqns, initial_guess, bounds=([0, 0], [1, 1])
            )
            # print(X, res.cost, res.x)
            return res.x

        def wittmer_eqns(X, fA):
            fA = fA[0] + 1e-40
            X += 1e-40
            A = fA * (1 - X) * (self.fA0 + self.fB0) * self.M0
            B = (1 - fA) * (1 - X) * (self.fA0 + self.fB0) * self.M0
            x1, y1 = wittmer_vars(X, A, B)

            dAdt = 1 + (r1 * A / B - r1 * K1 * (1 - x1) / B) / (
                1 - q1 * y1 / B * (B + q2 * x1) / (A + q1 * y1)
            )
            dBdt = 1 + (r2 * B / A - r2 * K2 * (1 - y1) / A) / (
                1 - q2 * x1 / A * (A + q1 * y1) / (B + q2 * x1)
            )
            FA = dAdt / (dAdt + dBdt)
            return [(fA - FA) / (1 - X)]

        sol = solve_ivp(
            wittmer_eqns,
            self.bounds,
            [self.fA0],
            method=self.method,
            max_step=0.001,
            t_eval=self.t_eval,
        )
        X = sol.t
        fA = sol.y[0]

        A = fA * (1 - X) * (self.fA0 + self.fB0)
        B = (1 - fA) * (1 - X) * (self.fA0 + self.fB0)

        xA = (self.fA0 - A) / (self.fA0)
        xB = (self.fB0 - B) / (self.fB0)

        return X, xA, xB

        # B_vals = np.linspace(0, 1, 1000)
        # A_vals = odeint(izu, [0], B_vals)

        # # Code here for solve the izu equation

        # # return results of integration
        # return B_vals, A_vals.flatten()

    def solve_kruger(self):

        rA = self.k1 / self.k3  # kAA / kAB
        rB = self.k7 / self.k5  # kBB / kBA
        RA = self.k4 / self.k5  # k_AB / kBA
        RAA = self.k2 / self.k3  # k_AA / kAB
        RB = self.k6 / self.k3  # k_BA / kAB
        RBB = self.k8 / self.k5  # k_BB / kBA

        # Express in terms of rA, rB, KAA, KAB, KBA, KBB

        # RA = k_AB / kBA = (k_AB/kAB)*(kAB/kAA)*(kBB/kBA) = KAB*(1/rA)*rB = k_AB/kBA * (kBB/kAA)
        # RAA = k_AA / kAB = (k_AA/kAA)*(kAA/kAB) = KAA*rA = k_AA/kAB
        # RB = k_BA / kAB = (k_BA/kBA)*(kBA/kBB)*(kAA/kAB) = KBA*(1/rB)*rA = k_BA/kAB * (kAA/kBB)
        # RBB = k_BB / kBA = (k_BB/kBB)*(kBB/kBA) = KBB*rB = k_BB/kBA

        def kruger_vars(X, A, B):
            def kruger_var_eqns(vars):
                PAA, PAB, PBA, PBB = vars
                a = 1 - RA * PBA / (A + RA * PBA)
                b = RAA - RA * RB * PBA / (A + RA * PBA)
                c = 1 - RB * PAB / (B + RB * PAB)
                d = RBB - RB * RA * PAB / (B + RB * PAB)

                eq1 = b * PAB * PAB + (rA * A + B * a - b) * PAB - B * a
                eq2 = d * PBA * PBA + (rB * B + A * c - d) * PBA - A * c
                eq3 = PAA + PAB - 1
                eq4 = PBB + PBA - 1

                return [eq1, eq2, eq3, eq4]

            initial_guess = [0.5, 0.5, 0.5, 0.5]
            res = least_squares(
                kruger_var_eqns, initial_guess, bounds=([0, 0, 0, 0], [1, 1, 1, 1])
            )
            # print(X, res.cost, res.x)
            return res.x

        # PAA + PAB + PBA + PBB = 1
        # fPAA = PAA / (PAA + PBA)
        # fPAB = PAB / (PAB + PBB)
        # fPBA = PBA / (PBA + PAA)
        # fPBB = PBB / (PBB + PAB)

        def kruger_eqns(X, fA):
            fA = fA[0] + 1e-40
            X += 1e-40
            A = fA * (1 - X) * (self.fA0 + self.fB0) * self.M0
            B = (1 - fA) * (1 - X) * (self.fA0 + self.fB0) * self.M0
            PAA, PAB, PBA, PBB = kruger_vars(X, A, B)

            dAdt = A * (rA * (A + RA * PBA) + B - RAA * PAA) - RA * PBA * (
                RAA * PAA + RB * PAB
            )
            dBdt = B * (rB * (B + RB * PAB) + A - RBB * PBB) - RB * PAB * (
                RBB * PBB + RA * PBA
            )
            FA = dAdt / (dAdt + dBdt)
            return [(fA - FA) / (1 - X)]

        sol = solve_ivp(
            kruger_eqns,
            self.bounds,
            [self.fA0],
            method=self.method,
            max_step=0.001,
            t_eval=self.t_eval,
        )

        X = sol.t
        fA = sol.y[0]

        A = fA * (1 - X) * (self.fA0 + self.fB0)
        B = (1 - fA) * (1 - X) * (self.fA0 + self.fB0)

        xA = (self.fA0 - A) / (self.fA0)
        xB = (self.fB0 - B) / (self.fB0)

        return X, xA, xB

import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.integrate import odeint, solve_ivp


class CPE:
    def __init__(self, inputs):
        self.fA0 = inputs[0]
        self.fB0 = 1 - self.fA0
        self.M0 = inputs[9]

        # Reaction rate constants, etc.
        self.kpAA = self.k1 = inputs[1]
        self.kdAA = self.k2 = inputs[2]
        self.kpAB = self.k3 = inputs[3]
        self.kdAB = self.k4 = inputs[4]
        self.kpBA = self.k5 = inputs[5]
        self.kdBA = self.k6 = inputs[6]
        self.kpBB = self.k7 = inputs[7]
        self.kdBB = self.k8 = inputs[8]

        self.bounds = [0, 1.0]
        self.num_points = 100
        self.t_eval = np.linspace(self.bounds[0], self.bounds[1], self.num_points)

        # Default method used by solve_ivp (can be overridden)
        self.method = "BDF"

    def set_bounds(self, bounds):
        self.bounds = bounds
        self.t_eval = np.linspace(self.bounds[0], self.bounds[1], self.num_points)

    @staticmethod
    def from_ratios(inputs, fA0=0.5, M0=1.0):
        """
        rA, rB, rX, KAA, KAB, KBA, KBB = inputs
        """
        rA, rB, rX, KAA, KAB, KBA, KBB = inputs
        kpAA = 1 # rA = kpAA/kpAB
        kpAB = kpAA / rA
        kpBB = kpAA / rX
        kpBA = kpBB / rB
        kdAA = KAA * kpAA
        kdAB = KAB * kpAB
        kdBA = KBA * kpBA
        kdBB = KBB * kpBB
        new_inputs = [fA0, kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB, M0]
        
        return CPE(new_inputs)

    @staticmethod
    def fromLundberg(inputs, M0=1.0):
        """
        Helper constructor if desired.
        """
        f1, r1, r2, b1, g1, b2, g2 = inputs
        new_inputs = [f1, 1, b1, 1 / r1, g2, 1 / r2, g1, 1, b2, M0]
        return CPE(new_inputs)

    # -------------------------------------------------------------------------
    # Private "vars" helper methods (systems of algebraic equations)
    # -------------------------------------------------------------------------
    def _izu_vars(self, X, A, B):
        """
        Solve the 'vars' system for the Izu approach.
        """

        def izu_var_eqns(vars):
            a, b, c, d = vars
            eq1 = a + b - 1
            eq2 = a * (self.k3 * B + (1 - c) * self.k6) - b * (
                self.k5 * A + (1 - d) * self.k4
            )
            eq3 = b * c * (1 - d) * self.k4 - a * (
                c * (self.k1 * A + self.k2 + self.k3 * B)
                - (self.k1 * A + c * c * self.k2)
            )
            eq4 = a * d * (1 - c) * self.k6 - b * (
                d * (self.k7 * B + self.k8 + self.k5 * A)
                - (self.k7 * B + d * d * self.k8)
            )
            return [eq1, eq2, eq3, eq4]

        initial_guess = [0.5, 0.5, 0.5, 0.5]
        res = least_squares(
            izu_var_eqns, initial_guess, bounds=([0, 0, 0, 0], [1, 1, 1, 1])
        )
        return res.x

    def _wittmer_vars(self, X, A, B):
        """
        Solve the 'vars' system for the Wittmer approach.
        """
        q1 = self.kdAB / self.kpBA
        q2 = self.kdBA / self.kpAB
        r1 = self.kpAA / self.kpAB
        r2 = self.kpBB / self.kpBA
        K1 = self.kdAA / self.kpAA
        K2 = self.kdBB / self.kpBB

        def wittmer_var_eqns(vars):
            x1, y1 = vars
            eq1 = (
                q1 * y1 * (B + q2 * x1) / (A + q1 * y1)
                - B
                - r1 * (A + K1)
                + 2 * r1 * (A + K1 * (1 - x1) ** 2) / (2 * (1 - x1))
            )
            eq2 = (
                q2 * x1 * (A + q1 * y1) / (B + q2 * x1)
                - A
                - r2 * (B + K2)
                + 2 * r2 * (B + K2 * (1 - y1) ** 2) / (2 * (1 - y1))
            )
            return [eq1, eq2]

        initial_guess = [0.5, 0.5]
        res = least_squares(wittmer_var_eqns, initial_guess, bounds=([0, 0], [1, 1]))
        return res.x

    def _kruger_vars(self, X, A, B):
        """
        Solve the 'vars' system for the Kruger approach.
        """
        # Some parameters for convenience
        rA = self.k1 / self.k3  # kAA / kAB
        rB = self.k7 / self.k5  # kBB / kBA
        RA = self.k4 / self.k5  # kdAB / kpBA
        RAA = self.k2 / self.k3  # kdAA / kpAB
        RB = self.k6 / self.k3  # kdBA / kpAB
        RBB = self.k8 / self.k5  # kdBB / kpBA

        def kruger_var_eqns(vars):
            PAA, PAB, PBA, PBB = vars
            a = 1 - RA * PBA / (A + RA * PBA)
            b = RAA - RA * RB * PBA / (A + RA * PBA)
            c = 1 - RB * PAB / (B + RB * PAB)
            d = RBB - RB * RA * PAB / (B + RB * PAB)

            eq1 = b * PAB * PAB + (rA * A + B * a - b) * PAB - B * a
            eq2 = d * PBA * PBA + (rB * B + A * c - d) * PBA - A * c
            eq3 = PAA + PAB - 1
            eq4 = PBB + PBA - 1
            return [eq1, eq2, eq3, eq4]

        initial_guess = [0.5, 0.5, 0.5, 0.5]
        res = least_squares(
            kruger_var_eqns, initial_guess, bounds=([0, 0, 0, 0], [1, 1, 1, 1])
        )
        return res.x

    # -------------------------------------------------------------------------
    # Private ODE systems for each approach
    # -------------------------------------------------------------------------
    def _izu_eqns(self, X, fA):
        """
        d(fA)/dX for the Izu approach.
        """
        fA_val = fA[0]
        A = fA_val * (1 - X) * (self.fA0 + self.fB0) * self.M0
        B = (1 - fA_val) * (1 - X) * (self.fA0 + self.fB0) * self.M0

        a, b, c, d = self._izu_vars(X, A, B)

        dAdt = a * self.k1 * A + b * self.k5 * A - a * ((1 - c) * self.k6 + c * self.k2)
        dBdt = b * self.k7 * B + a * self.k3 * B - b * ((1 - d) * self.k4 + d * self.k8)

        FA = dAdt / (dAdt + dBdt)
        dfA_dX = (fA_val - FA) / (1 - X)
        return [dfA_dX]

    def _wittmer_eqns(self, X, fA):
        """
        d(fA)/dX for the Wittmer approach.
        """
        fA_val = fA[0] + 1e-40
        X_val = X + 1e-40

        A = fA_val * (1 - X_val) * (self.fA0 + self.fB0) * self.M0
        B = (1 - fA_val) * (1 - X_val) * (self.fA0 + self.fB0) * self.M0

        # Solve for the 'vars' needed
        x1, y1 = self._wittmer_vars(X_val, A, B)

        q1 = self.kdAB / self.kpBA
        q2 = self.kdBA / self.kpAB
        r1 = self.kpAA / self.kpAB
        r2 = self.kpBB / self.kpBA
        K1 = self.kdAA / self.kpAA
        K2 = self.kdBB / self.kpBB

        dAdt = 1 + (r1 * A / B - r1 * K1 * (1 - x1) / B) / (
            1 - q1 * y1 / B * (B + q2 * x1) / (A + q1 * y1)
        )
        dBdt = 1 + (r2 * B / A - r2 * K2 * (1 - y1) / A) / (
            1 - q2 * x1 / A * (A + q1 * y1) / (B + q2 * x1)
        )

        FA = dAdt / (dAdt + dBdt)
        return [(fA_val - FA) / (1 - X_val)]

    def _kruger_eqns(self, X, fA):
        """
        d(fA)/dX for the Kruger approach.
        """
        fA_val = fA[0] + 1e-40
        X_val = X + 1e-40

        A = fA_val * (1 - X_val) * (self.fA0 + self.fB0) * self.M0
        B = (1 - fA_val) * (1 - X_val) * (self.fA0 + self.fB0) * self.M0

        PAA, PAB, PBA, PBB = self._kruger_vars(X_val, A, B)

        # Some parameters for convenience
        rA = self.k1 / self.k3  # kAA / kAB
        rB = self.k7 / self.k5  # kBB / kBA
        RA = self.k4 / self.k5  # kdAB / kpBA
        RAA = self.k2 / self.k3  # kdAA / kpAB
        RB = self.k6 / self.k3  # kdBA / kpAB
        RBB = self.k8 / self.k5  # kdBB / kpBA

        dAdt = A * (rA * (A + RA * PBA) + B - RAA * PAA) - RA * PBA * (
            RAA * PAA + RB * PAB
        )
        dBdt = B * (rB * (B + RB * PAB) + A - RBB * PBB) - RB * PAB * (
            RBB * PBB + RA * PBA
        )

        FA = dAdt / (dAdt + dBdt)
        return [(fA_val - FA) / (1 - X_val)]

    # -------------------------------------------------------------------------
    # Helper for post-processing the solution
    # -------------------------------------------------------------------------
    def _compute_conversion(self, X, fA):
        """
        Given arrays X and fA, compute xA and xB conversions
        and return (X, xA, xB).
        """
        A = fA * (1 - X) * (self.fA0 + self.fB0)
        B = (1 - fA) * (1 - X) * (self.fA0 + self.fB0)

        xA = (self.fA0 - A) / (self.fA0)
        xB = (self.fB0 - B) / (self.fB0)
        return X, xA, xB

    # -------------------------------------------------------------------------
    # Single public solve method
    # -------------------------------------------------------------------------
    def solve(self, method_name="izu", t_eval=None, **kwargs):
        """
        Solve the copolymer system using one of the approaches:
          - "izu"
          - "wittmer"
          - "kruger"

        Parameters
        ----------
        method_name : str
            Which approach to use. Valid values: "izu", "wittmer", "kruger".
        t_eval : array-like or None
            The time (or conversion) grid on which to store the solution.
            If None, uses the default self.t_eval.
        **kwargs :
            Additional arguments passed to solve_ivp(), e.g. max_step, rtol, atol, etc.

        Returns
        -------
        X : np.ndarray
            The conversion array (sol.t).
        xA : np.ndarray
            Conversion of species A.
        xB : np.ndarray
            Conversion of species B.
        """
        # Use default t_eval if not explicitly passed
        if t_eval is None:
            t_eval = self.t_eval

        # If the user didn't specify method=..., use self.method.
        solver_method = kwargs.pop("method", self.method)

        # Dispatch to the correct ODE system
        if method_name == "izu":
            ode_func = self._izu_eqns
        elif method_name == "wittmer":
            ode_func = self._wittmer_eqns
        elif method_name == "kruger":
            ode_func = self._kruger_eqns
        else:
            raise ValueError(
                f"Unknown method '{method_name}'. "
                "Choose from 'izu', 'wittmer', or 'kruger'."
            )

        # We can also set a default max_step if desired:
        kwargs.setdefault("max_step", 0.001)

        # Solve the IVP
        sol = solve_ivp(
            fun=ode_func,
            t_span=self.bounds,
            y0=[self.fA0],
            method=solver_method,
            t_eval=t_eval,
            **kwargs,
        )

        # Unpack solution
        X = sol.t
        fA = sol.y[0]

        # Post-process to get xA, xB
        return self._compute_conversion(X, fA)
