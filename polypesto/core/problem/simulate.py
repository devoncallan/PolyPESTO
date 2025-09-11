from typing import List, Tuple
from pathlib import Path


from polypesto.models import ModelBase
from .. import petab as pet
from ..params import ParameterSet
from ..conditions import SimConditions, conditions_to_df
from ..problem import PypestoProblem, Problem, write_petab


def write_empty_problem(
    data_dir: str | Path,
    model: ModelBase,
    conds: List[SimConditions],
) -> Tuple[Problem, ParameterSet]:
    """Create an empty problem and parameter set.

    Args:
        data_dir (str | Path): Directory where the data is stored.
        model (ModelBase): The model to be used for simulation.
        conds (List[SimConditions]): List of simulation conditions for each experiment.

    Returns:
        Tuple[Problem, ParameterSet]: An empty problem (no measurements) and the true parameters.
    """

    data_dict = {
        (f"obs_{obs_id}", cond.exp_id): cond.t_eval
        for cond in conds
        for obs_id in model.observables.keys()
    }

    petab_data = pet.PetabData(
        obs_df=model.get_obs_df(),
        cond_df=conditions_to_df(conds),
        param_df=model.get_param_df(),
        meas_df=pet.define_empty_measurements(data_dict),
    )

    true_params = conds[0].true_params
    problem = write_petab(data_dir, model, petab_data, true_params)

    return problem, true_params


def simulate_experiments(
    data_dir: str | Path,
    model: ModelBase,
    conds: List[SimConditions],
) -> Problem:
    """Simulate experiments based on the provided conditions.

    Args:
        data_dir (str | Path): Directory where the data is stored.
        model (ModelBase): The model to be used for simulation.
        conds (List[SimConditions]): List of simulation conditions for each experiment.

    Returns:
        Problem: The problem instance containing the simulation results.
    """

    from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df
    from pypesto.objective import AmiciObjective

    problem, true_params = write_empty_problem(data_dir, model, conds)

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

    pet.PetabIO.write_meas_df(meas_df, filename=problem.paths.measurements)

    return Problem.load(
        model=problem.model,
        paths=problem.paths,
    )
