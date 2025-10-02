from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from amici.petab.simulations import (  # type: ignore
    rdatas_to_measurement_df,
    simulate_petab,
)
from numpy.typing import ArrayLike
from pypesto.objective import AmiciObjective  # type: ignore

from polypesto.models import ModelBase
from polypesto.utils import ID, read_json, write_json

from .. import petab as pet
from ..params import ParameterSet
from ..pypesto import PypestoProblem
from .core import ProblemPaths
from .problem import Problem, write_petab


@dataclass
class SimConditions:
    """Conditions for the simulation."""

    true_params: ParameterSet
    conds: ParameterSet
    t_eval: np.ndarray
    noise_level: float = 0.0


def write_sim_conditions(
    paths: ProblemPaths,
    sim_conditions: Sequence[SimConditions],
) -> ParameterSet:
    """Write a list of SimConditions to a JSON file."""
    cond_ids = [sim_cond.conds.id for sim_cond in sim_conditions]

    sim_conds_dict = {}
    param_set: Dict[str, ParameterSet] = {}
    for cond_id, sim_cond in zip(cond_ids, sim_conditions, strict=True):
        param_set[cond_id] = sim_cond.true_params
        sim_conds_dict[cond_id] = {
            "conds": sim_cond.conds.to_dict(),
            "t_eval": sim_cond.t_eval.tolist(),
            "noise_level": sim_cond.noise_level,
        }
    # Check that all true_params are the same
    true_params_list = list(param_set.values())
    if not all(tp == true_params_list[0] for tp in true_params_list):
        raise ValueError("All SimConditions must have the same true_params.")
    true_params = true_params_list[0]

    output_dict: Dict[str, Any] = {}
    output_dict["sim_conds"] = sim_conds_dict
    output_dict["true_params"] = true_params.to_dict()

    write_json(paths.sim_conds, output_dict)

    return true_params


def load_sim_conditions(
    paths: ProblemPaths,
) -> Tuple[ParameterSet, List[SimConditions]]:
    """Load a list of SimConditions from a JSON file."""

    data = read_json(paths.sim_conds)
    true_params = ParameterSet.from_dict(data.pop("true_params"))

    sim_conds = []
    data = data["sim_conds"]
    for cond_id, sim_cond_data in data.items():
        sim_cond = SimConditions(
            true_params=true_params,
            conds=ParameterSet.from_dict(sim_cond_data["conds"], id=cond_id),
            t_eval=np.array(sim_cond_data["t_eval"]),
            noise_level=float(sim_cond_data["noise_level"]),
        )
        sim_conds.append(sim_cond)

    return true_params, sim_conds


@dataclass
class SimulatedProblem(Problem):
    """A parameter estimation problem with simulated data."""

    true_params: ParameterSet
    sim_conditions: List[SimConditions]

    @staticmethod
    def from_problem(
        problem: Problem, true_params: ParameterSet, sim_conditions: List[SimConditions]
    ) -> SimulatedProblem:
        return SimulatedProblem(
            model=problem.model,
            petab_problem=problem.petab_problem,
            pypesto_problem=problem.pypesto_problem,
            paths=problem.paths,
            experiments=problem.experiments,
            true_params=true_params,
            sim_conditions=sim_conditions,
        )

    @staticmethod
    def load(prob_dir: str | Path, model: ModelBase, **kwargs) -> SimulatedProblem:

        problem = Problem.load(prob_dir, model, **kwargs)
        true_params, sim_conditions = load_sim_conditions(problem.paths)
        return SimulatedProblem.from_problem(problem, true_params, sim_conditions)

    def visualize_results(self, **kwargs):
        true_params = self.true_params.to_dict()
        return super().visualize_results(true_params=true_params, **kwargs)


def create_sim_conditions(
    conds: Mapping[str, ArrayLike],
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

    Returns:
        List[SimConditions]: List of simulation conditions."""

    if isinstance(true_params, dict):
        true_params = ParameterSet.from_dict(true_params)
    elif not isinstance(true_params, ParameterSet):
        raise ValueError("true_params must be a ParameterSet or a dict.")

    conds_list = ParameterSet.from_dict_list(conds)
    cond_ids = ID.make_cond_ids(len(conds_list))
    conds_list = [
        cond.set_id(cond_id) for cond, cond_id in zip(conds_list, cond_ids, strict=True)
    ]

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
    sim_conds: List[SimConditions],
) -> SimulatedProblem:
    """Create an empty problem and parameter set.

    Args:
        prob_dir (str | Path): Directory where the data is stored.
        model (ModelBase): The model to be used for simulation.
        conds (List[SimConditions]): List of simulation conditions for each experiment.

    Returns:
        SimulatedProblem: An empty problem (no measurements).
    """

    data_dict = {
        (ID.obs_id(obs_name), sim_cond.conds.id): sim_cond.t_eval
        for sim_cond in sim_conds
        for obs_name in model.observables.keys()
    }
    conds_list = [cond.conds.to_dict() for cond in sim_conds]
    cond_ids = [cond.conds.id for cond in sim_conds]

    petab_data = pet.PetabData(
        obs_df=model.get_obs_df(),
        cond_df=pet.define_conditions(conds_list, ids=cond_ids),
        param_df=model.get_param_df(),
        meas_df=pet.define_empty_measurements(data_dict),
    )

    problem = write_petab(prob_dir, model, petab_data)
    true_params = write_sim_conditions(problem.paths, sim_conds)

    problem = SimulatedProblem.from_problem(problem, true_params, sim_conds)

    return problem


def simulate_problem(
    prob_dir: str | Path,
    model: ModelBase,
    conds: List[SimConditions],
    overwrite: bool = False,
) -> SimulatedProblem:
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
            problem = SimulatedProblem.load(prob_dir, model)
            # TODO: Check that conditions from loaded problem match provided conditions
            print("Successfully loaded existing problem.")
            return problem
        except Exception as e:
            print(f"Failed to load problem: {e}")
            print("Proceeding to simulate new data.")

    problem = write_empty_problem(prob_dir, model, conds)

    pypesto_problem = problem.pypesto_problem
    petab_problem = problem.petab_problem

    assert isinstance(pypesto_problem, PypestoProblem)
    assert isinstance(pypesto_problem.objective, AmiciObjective)

    # Simulate experiment
    sim_data = simulate_petab(
        petab_problem=petab_problem,
        amici_model=pypesto_problem.objective.amici_model,
        solver=pypesto_problem.objective.amici_solver,
        problem_parameters=problem.true_params.to_dict(),
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

    return SimulatedProblem.load(
        prob_dir=prob_dir,
        model=problem.model,
    )
