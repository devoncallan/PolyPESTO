from typing import List, Tuple
from pathlib import Path


from polypesto.models import ModelBase
from .. import petab as pet
from ..params import ParameterSet
from ..conditions import SimConditions, conditions_to_df
from ..problem import PypestoProblem, Problem, write_petab


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
            print("Successfully loaded existing problem.")
            return problem
        except Exception as e:
            print(f"Failed to load problem: {e}")
            print("Proceeding to simulate new data.")

    from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df
    from pypesto.objective import AmiciObjective

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

    pet.PetabIO.write_meas_df(meas_df, filename=problem.paths.measurements)

    return Problem.load(
        prob_dir=prob_dir,
        model=problem.model,
    )
