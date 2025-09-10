from typing import List
from pathlib import Path

from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df
from pypesto.objective import AmiciObjective

from polypesto.models import ModelBase
from . import petab as pet
from .params import ParameterSet
from .conditions import SimConditions, conditions_to_df
from .problem import Problem, ProblemPaths, write_petab
from .pypesto import load_pypesto_problem, PypestoProblem


def _simulate_from_empty_problem(
    empty_prob: Problem,
    true_params: ParameterSet,
    **kwargs,
) -> Problem:

    importer, pypesto_problem = load_pypesto_problem(
        yaml_path=empty_prob.paths.petab_yaml,
        model_name=empty_prob.model.name,
        **kwargs,
    )
    petab_problem = importer.petab_problem

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
        meas_df, noise_level=empty_prob.model.obs_noise_level
    )

    pet.PetabIO.write_meas_df(meas_df, filename=empty_prob.paths.measurements)

    return Problem.load(
        model=empty_prob.model,
        paths=empty_prob.paths,
    )


def simulate_experiments(
    data_dir: str | Path,
    model: ModelBase,
    conds: List[SimConditions],
) -> Problem:

    obs_df = model.get_obs_df()
    param_df = model.get_param_df()
    cond_df = conditions_to_df(conds)

    data_dict = {}
    for cond in conds:
        for obs in model.observables.keys():
            obs_id = f"obs_{obs}"
            data_dict[(obs_id, cond.exp_id)] = cond.t_eval
    meas_df = pet.define_empty_measurements(data_dict)

    petab_data = pet.PetabData(obs_df, cond_df, param_df, meas_df)
    paths = ProblemPaths(data_dir)

    true_params = conds[0].true_params
    write_petab(model, paths, petab_data, true_params)

    problem = Problem.load(model, paths)
    problem = _simulate_from_empty_problem(problem, true_params)

    return problem
