import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt


def irreversible_penultimate_model(rA=2.0, rB=0.5, fA0=0.5, verbose=True):
    """
    Robust solver for irreversible penultimate copolymerization.

    Parameters:
    - rA, rB: Reactivity ratios
    - fA0: Initial mole fraction of A
    """
    # Set up parameters
    kAB = 1.0  # Reference rate constant
    kAA = rA * kAB
    kBB = rB * kAB
    kBA = kBB / rB

    if verbose:
        print(f"Parameters: rA={rA}, rB={rB}, fA0={fA0}")
        print(f"Rate constants: kAA={kAA}, kAB={kAB}, kBA={kBA}, kBB={kBB}")

    # Initial conditions
    cM0 = 1.0  # Total monomer concentration
    A0 = fA0 * cM0
    B0 = (1 - fA0) * cM0

    # Function to solve algebraic constraints - FIXED SHAPE ISSUE
    def solve_chain_end_probabilities(A, B, p_init=None):
        """Solve for chain-end probabilities with robust error handling."""
        if p_init is None:
            p_init = np.array([0.5, 0.5, 0.5])  # pA, pAA, pAB

        def equations(p):
            pA, pAA, pAB = p
            pB = 1 - pA
            pBA = 1 - pAA
            pBB = 1 - pAB

            # # Return scalars, not arrays
            # eq1 = float(kAA * pBA * pA * A - kAB * pAA * pA * B)
            # eq2 = float(kBB * pAB * pB * B - kBA * pBB * pB * A)
            # eq3 = float(kAB * pA * B - pAB * pB * (kBA * A + kBB * B))

            # Small regularization to prevent division by zero
            eps = 1e-10

            # Use relative rather than absolute differences
            eq1 = (kAA * pBA * A - kAB * pAA * B) / (abs(kAA * pBA * A) + abs(kAB * pAA * B) + eps)
            eq2 = (kBB * pAB * B - kBA * pBB * A) / (abs(kBB * pAB * B) + abs(kBA * pBB * A) + eps) 
            eq3 = (kAB * pA * B - pAB * pB * (kBA * A + kBB * B)) / (abs(kAB * pA * B) + abs(pAB * pB * (kBA * A + kBB * B)) + eps)

            return np.array([eq1, eq2, eq3])

        # Try different solvers
        methods = ["hybr", "lm"]
        for method in methods:
            try:
                result = root(equations, p_init, method=method)

                if result.success:
                    pA, pAA, pAB = result.x
                    pB = 1 - pA
                    pBA = 1 - pAA
                    pBB = 1 - pAB

                    # Validate results
                    if (
                        min(pA, pB, pAA, pAB, pBA, pBB) >= 0
                        and max(pA, pB, pAA, pAB, pBA, pBB) <= 1
                    ):
                        return pA, pB, pAA, pAB, pBA, pBB, True

            except Exception as e:
                if verbose:
                    print(f"Solver {method} failed: {e}")

        # If all methods fail, use a fallback approach
        if verbose:
            print(f"All solvers failed at A={A:.6f}, B={B:.6f}, using fallback")

        # Fallback to simpler estimate
        pA = A / (A + B)
        pB = 1 - pA
        pAA = pA
        pAB = pA
        pBA = pB
        pBB = pB

        return pA, pB, pAA, pAB, pBA, pBB, False

    # ODE system for dxA/dX
    def dxA_dX(X, xA):
        """Calculate dxA/dX with robust error handling."""
        # Calculate current concentrations
        A = A0 * (1 - xA[0])  # xA is now array from solve_ivp
        total_X = X * (A0 + B0)
        B = B0 - (total_X - A0 * xA[0])
        xB = 1 - B / B0

        # Prevent negative or zero concentrations
        A = max(A, 1e-10)
        B = max(B, 1e-10)

        # Solve for chain-end probabilities
        pA, pB, pAA, pAB, pBA, pBB, success = solve_chain_end_probabilities(A, B)

        # Calculate monomer consumption rates
        rA_rate = A * (kAA * pA + kBA * pB)
        rB_rate = B * (kBB * pB + kAB * pA)

        # Calculate dxA/dX with safety check for division by zero
        denominator = rA_rate + rB_rate
        if denominator < 1e-10:
            denominator = 1e-10

        result = (A0 + B0) / A0 * rA_rate / denominator

        dxA_dx = result
        dxB_dx = (A0 + B0) / B0 * rB_rate / denominator

        # Periodic diagnostic output
        if verbose and (round(X * 10) % 1 < 0.001 or round(X * 10) % 1 > 0.999):
            print(f"At X={X:.2f}, xA={xA[0]:.3f}, xB = {xB:.3f}, A={A:.4f}, B={B:.4f}")
            print(f"  pA={pA:.4f}, pB={pB:.4f}")
            print(f"  pAA={pAA:.4f}, pBA={pBA:.4f}")
            print(f"  pAB={pAB:.4f}, pBB={pBB:.4f}")
            print(f"  dxA/dX={dxA_dx:.4f}")
            print(f"  dxB/dX={dxB_dx:.4f}")

        return [result]  # Return as array for solve_ivp

    # Event function to stop integration when needed
    def stop_condition(X, xA):
        A = A0 * (1 - xA[0])
        total_X = X * (A0 + B0)
        B = B0 - (total_X - A0 * xA[0])

        if A < 0.01 * A0 and B < 0.01 * B0 or X > 0.99:
            return 0
        return 1

    stop_condition.terminal = True

    # Run the simulation
    if verbose:
        print("\nStarting simulation...")

    # Initial condition
    xA0 = [0.0]  # As array for solve_ivp

    try:
        solution = solve_ivp(
            dxA_dX,
            [0, 0.99],  # X range
            xA0,
            method="BDF",  # Good for stiff problems
            events=stop_condition,
            rtol=1e-3,
            atol=1e-5,
            max_step=0.05,
            dense_output=True,
        )

        if verbose:
            print(f"Integration completed: {solution.message}")
            print(f"Integration reached X={solution.t[-1]:.4f}")

        # Process results
        X_eval = np.linspace(0, solution.t[-1], 100)
        xA_eval = solution.sol(X_eval)[0]

        # Calculate derived quantities
        results = {
            "X": X_eval,
            "xA": xA_eval,
            "xB": (X_eval * (A0 + B0) - xA_eval * A0) / B0,
            "A": A0 * (1 - xA_eval),
            "B": B0 * (1 - ((X_eval * (A0 + B0) - xA_eval * A0) / B0)),
            "pA": np.zeros_like(X_eval),
            "pB": np.zeros_like(X_eval),
            "pAA": np.zeros_like(X_eval),
            "pAB": np.zeros_like(X_eval),
            "pBA": np.zeros_like(X_eval),
            "pBB": np.zeros_like(X_eval),
        }

        # Calculate chain-end probabilities at each point
        for i, (X, xA) in enumerate(zip(X_eval, xA_eval)):
            A = results["A"][i]
            B = results["B"][i]
            # Skip chain-end probability calculation for very small concentrations
            if A < 1e-8 or B < 1e-8:
                results["pA"][i] = 0.5
                results["pB"][i] = 0.5
                results["pAA"][i] = 0.5
                results["pAB"][i] = 0.5
                results["pBA"][i] = 0.5
                results["pBB"][i] = 0.5
            else:
                pA, pB, pAA, pAB, pBA, pBB, _ = solve_chain_end_probabilities(A, B)
                results["pA"][i] = pA
                results["pB"][i] = pB
                results["pAA"][i] = pAA
                results["pAB"][i] = pAB
                results["pBA"][i] = pBA
                results["pBB"][i] = pBB

        if verbose:
            print("\nSimulation successful!")

        return results

    except Exception as e:
        if verbose:
            print(f"Simulation failed: {str(e)}")
        return None


# Function to plot results
def plot_results(results):
    if results is None:
        print("No results to plot.")
        return

    plt.figure(figsize=(12, 10))

    # Conversion plot
    plt.subplot(2, 2, 1)
    plt.plot(results["X"], results["xA"], "b-", label="A Conversion")
    plt.plot(results["X"], results["xB"], "r-", label="B Conversion")
    plt.xlabel("Overall Conversion (X)")
    plt.ylabel("Individual Conversion")
    plt.legend()
    plt.grid(True)
    plt.title("Monomer Conversion")

    # Terminal probabilities
    plt.subplot(2, 2, 2)
    plt.plot(results["X"], results["pA"], "b-", label="pA")
    plt.plot(results["X"], results["pB"], "r-", label="pB")
    plt.xlabel("Overall Conversion (X)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.title("Terminal Chain-end Probabilities")

    # Penultimate probabilities
    plt.subplot(2, 2, 3)
    plt.plot(results["X"], results["pAA"], "b-", label="pAA")
    plt.plot(results["X"], results["pBA"], "g-", label="pBA")
    plt.xlabel("Overall Conversion (X)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.title("A-Terminal Penultimate Probabilities")

    plt.subplot(2, 2, 4)
    plt.plot(results["X"], results["pBB"], "r-", label="pBB")
    plt.plot(results["X"], results["pAB"], "g-", label="pAB")
    plt.xlabel("Overall Conversion (X)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.title("B-Terminal Penultimate Probabilities")

    plt.tight_layout()
    plt.show()


# Run a simulation with a well-behaved case
def run_simple_case():
    print("Running well-behaved case (rA=rB=1.0)...")
    results = irreversible_penultimate_model(rA=1.0, rB=1.0, fA0=0.5, verbose=True)
    if results:
        plot_results(results)
    else:
        print("Simulation failed.")


# Run a few test cases covering your parameter space
def run_test_cases():
    cases = [
        # {"rA": 0.1, "rB": 0.1, "fA0": 0.5},
        # {"rA": 1.0, "rB": 1.0, "fA0": 0.5},
        {"rA": 0.1, "rB": 10.0, "fA0": 0.5},
    ]

    for i, case in enumerate(cases):
        print(f"\n===== Test Case {i+1}: rA={case['rA']}, rB={case['rB']} =====")
        results = irreversible_penultimate_model(**case)
        if results:
            plot_results(results)


# Run the test cases
run_test_cases()


# # Run the simple case
# run_simple_case()
