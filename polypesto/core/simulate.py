import os
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Type, TypeVar

from numpy.typing import ArrayLike
import pandas as pd

from polypesto.core.params import ParameterSet
from polypesto.core.problem import ProblemPaths, Problem, write_and_load_problem
from polypesto.models.base import ModelBase
from polypesto.core.petab import (
    PetabData,
    define_empty_measurements,
    PetabIO,
    add_noise_to_measurements,
)

from polypesto.core.experiment import SimConditions


def simulate_problem(
    problem: Problem,
    true_params: ParameterSet,
    **kwargs,
) -> Problem:

    from polypesto.core.pypesto import load_pypesto_problem, PypestoProblem
    from polypesto.core.experiment import create_experiments_from_petab
    from pypesto.objective import AmiciObjective
    from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df

    importer, pypesto_problem = load_pypesto_problem(
        yaml_path=problem.paths.petab_yaml, model_name=problem.model.name, **kwargs
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

    meas_df = add_noise_to_measurements(
        meas_df, noise_level=problem.model.obs_noise_level
    )

    PetabIO.write_meas_df(meas_df, filename=problem.paths.measurements)

    return Problem.load(
        model=problem.model,
        paths=problem.paths,
    )


def simulate_experiments(
    data_dir: str,
    model: ModelBase,
    conds: List[SimConditions],
) -> Problem:

    from polypesto.core.experiment import define_cond_df

    cond_df = define_cond_df(conds)
    obs_df = model.get_obs_df()
    param_df = model.get_param_df()
    data_dict = {}
    for cond in conds:
        for obs in model.observables.keys():
            obs_id = f"obs_{obs}"
            data_dict[(obs_id, cond.exp_id)] = cond.t_eval
    meas_df = define_empty_measurements(data_dict)

    petab_data = PetabData(obs_df, cond_df, param_df, meas_df)
    paths = ProblemPaths(data_dir)

    true_params = conds[0].true_params
    problem = write_and_load_problem(model, paths, petab_data, true_params)

    problem = simulate_problem(problem, true_params)

    return problem