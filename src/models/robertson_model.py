import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def robertson_odes(t, y, k):
    """
    ODE system for the Robertson problem.

    Parameters
    ----------
    t : float
        Independent variable (time).
    y : array-like
        Dependent variables [y1, y2, y3].
    k : Tuple[float]
        Reaction rate constants [k1, k2, k3].

    Returns
    -------
    dydt : list
        Derivatives [dy1/dt, dy2/dt, dy3/dt].
    """
    y1, y2, y3 = y
    k1, k2, k3 = k

    dy1_dt = -k1 * y1 + k2 * y2 * y3
    dy2_dt = k1 * y1 - k2 * y2 * y3 - k3 * y2**2
    dy3_dt = k3 * y2**2

    return [dy1_dt, dy2_dt, dy3_dt]


def main():
    # --- 1. Define parameters (k1, k2, k3) ---
    k1 = 0.04
    k2 = 1.0e4
    k3 = 3.0e7
    k = [k1, k2, k3]

    # --- 2. Initial conditions and time span ---
    y0 = [1.0, 0.0, 0.0]  # y1(0)=1, y2(0)=0, y3(0)=0
    t_span = (0, 1e7)  # Integrate from t=0 to t=1e5

    # --- 3. Solve using solve_ivp for a stiff system (BDF or Radau) ---
    sol = solve_ivp(
        fun=robertson_odes,
        t_span=t_span,
        y0=y0,
        method="Radau",  # or 'Radau'
        dense_output=True,
        args=[k],  # pass in the reaction rate constants
    )

    # --- 4. Evaluate the solution on a convenient time grid ---
    t_eval = np.logspace(-5, 7, 400)  # from 1e-2 to 1e5 on a log scale
    y_eval = sol.sol(t_eval)

    y1_eval, y2_eval, y3_eval = y_eval
    print(y1_eval)

    # --- 5. Plot the results ---
    plt.figure(figsize=(8, 5))
    plt.plot(t_eval, y1_eval, label="y1")
    plt.plot(t_eval, y2_eval, label="y2")
    plt.plot(t_eval, y3_eval, label="y3")
    plt.xscale("log")  # log scale for time
    # plt.yscale("log")  # log scale for solution variables
    plt.xlabel("Time t")
    plt.ylabel("Concentration")
    plt.title("Robertson Problem (k1=%.2e, k2=%.1e, k3=%.1e)" % (k1, k2, k3))
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
