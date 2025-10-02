from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df  # type: ignore
from pypesto.objective import AmiciObjective  # type: ignore

from polypesto.models import ModelBase
from .. import petab as pet
from ..params import ParameterSet

from ..problem import Problem, write_petab
from ..pypesto import PypestoProblem


@dataclass
class SimConditions:
    """Conditions for the simulation."""

    true_params: ParameterSet
    conds: ParameterSet
    t_eval: np.ndarray
    noise_level: float = 0.0


def create_sim_conditions(
    conds: Dict[str, ArrayLike],
    true_params: ParameterSet | Dict[str, float],
    t_evals: ArrayLike | List[ArrayLike],
    noise_levels: float | List[float] = 0.0,
) -> List[SimConditions]:
    """Create a list of SimConditions from the provided parameters.

    Args:
        true_params (ParameterSet | Dict[str, float]): True parameter values.
            e.g., `true_params = {"rA": 0.25, "rB": 0.75}`
        conds (Dict[str, ArrayLike]): Dictionary of condition values.
            e.g., `conds = {"A0": [0.25, 0.6], "B0": [0.75, 0.4]}`
        t_evals (ArrayLike | List[ArrayLike]): Time evaluation points.
            e.g., `t_evals = np.linspace(0, 10, 100)` or `t_evals = [np.linspace(0, 10, 100), np.linspace(0, 5, 50)]`
        noise_levels (float | List[float]): Noise levels for the simulations. Defaults to 0.0.
            e.g., `noise_levels = 0.1` or `noise_levels = [0.1, 0.2]`
        exp_ids (Optional[List[str]]): List of experiment IDs. Defaults to None.
            e.g., `exp_ids = ["c_0", "c_1"]`. If None, will be auto-generated.

    Returns:
        List[SimConditions]: List of simulation conditions."""

    if isinstance(true_params, dict):
        true_params = ParameterSet.from_dict(true_params)
    elif not isinstance(true_params, ParameterSet):
        raise ValueError("true_params must be a ParameterSet or a dict.")

    conds_list = ParameterSet.from_dict_list(conds)
    n_conds = len(conds_list)

    if isinstance(t_evals, np.ndarray):
        t_evals_list = [np.array(t_evals)] * n_conds
    elif isinstance(t_evals, list):
        if len(t_evals) != n_conds:
            raise ValueError(
                f"Length of t_evals ({len(t_evals)}) must match number of conditions ({n_conds})."
            )
        t_evals_list = [np.array(te) for te in t_evals]
    else:
        raise TypeError(
            "t_evals must be convertible to a numpy array or a list of such."
        )
    assert isinstance(t_evals_list, list) and len(t_evals_list) == n_conds

    if isinstance(noise_levels, float):
        noise_levels_list = [noise_levels] * n_conds
    elif isinstance(noise_levels, list):
        if len(noise_levels) != n_conds:
            raise ValueError(
                f"Length of noise_levels ({len(noise_levels)}) must match number of conditions ({n_conds})."
            )
        noise_levels_list = noise_levels
    else:
        raise TypeError("noise_levels must be a float or a list of floats.")
    assert isinstance(noise_levels_list, list) and len(noise_levels_list) == n_conds

    sim_conditions = []
    for i in range(n_conds):

        sim_cond = SimConditions(
            true_params=true_params,
            conds=conds_list[i],
            t_eval=t_evals_list[i],
            noise_level=noise_levels_list[i],
        )
        sim_conditions.append(sim_cond)

    return sim_conditions


def write_empty_problem(
    prob_dir: str | Path,
    model: ModelBase,
    conds: List[SimConditions],
) -> Tuple[Problem, ParameterSet]:
    """Create an empty problem and parameter set.

    Args:
        prob_dir (str | Path): Directory where the data is stored.
        model (ModelBase): The model to be used for simulation.
        conds (List[SimConditions]): List of simulation conditions for each experiment.

    Returns:
        Tuple[Problem, ParameterSet]: An empty problem (no measurements) and the true parameters.
    """

    data_dict = {
        (f"obs_{obs_id}", cond.conds.id): cond.t_eval
        for cond in conds
        for obs_id in model.observables.keys()
    }
    conds_list = [cond.conds.to_dict() for cond in conds]

    petab_data = pet.PetabData(
        obs_df=model.get_obs_df(),
        cond_df=pet.define_conditions(conds_list),
        param_df=model.get_param_df(),
        meas_df=pet.define_empty_measurements(data_dict),
    )

    true_params = conds[0].true_params
    problem = write_petab(prob_dir, model, petab_data, true_params)

    return problem, true_params


def simulate_problem(
    prob_dir: str | Path,
    model: ModelBase,
    conds: List[SimConditions],
    overwrite: bool = False,
) -> Problem:
    """Simulate experiments based on the provided conditions.

    Args:
        prob_dir (str | Path): Directory where the data is stored.
        model (ModelBase): The model to be used for simulation.
        conds (List[SimConditions]): List of simulation conditions for each experiment.
        overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.

    Returns:
        Problem: The problem instance containing the simulation results.
    """

    if not overwrite and Path(prob_dir).exists():
        print(f"Data directory {prob_dir} already exists. Attempting to load problem.")
        try:
            problem = Problem.load(prob_dir, model)
            # TODO: Check that conditions from loaded problem match provided conditions
            print("Successfully loaded existing problem.")
            return problem
        except Exception as e:
            print(f"Failed to load problem: {e}")
            print("Proceeding to simulate new data.")

    problem, true_params = write_empty_problem(prob_dir, model, conds)

    pypesto_problem = problem.pypesto_problem
    petab_problem = problem.petab_problem

    assert isinstance(pypesto_problem, PypestoProblem)
    assert isinstance(pypesto_problem.objective, AmiciObjective)

    # Simulate experiment
    sim_data = simulate_petab(
        petab_problem=petab_problem,
        amici_model=pypesto_problem.objective.amici_model,
        solver=pypesto_problem.objective.amici_solver,
        problem_parameters=true_params.to_dict(),
    )

    # Create measurement DataFrame
    meas_df = rdatas_to_measurement_df(
        sim_data["rdatas"],
        pypesto_problem.objective.amici_model,
        petab_problem.measurement_df,
    )

    meas_df = pet.add_noise_to_measurements(
        meas_df, noise_level=problem.model.obs_noise_level
    )

    pet.write_measurement_df(meas_df, problem.paths.measurements)

    return Problem.load(
        prob_dir=prob_dir,
        model=problem.model,
    )
