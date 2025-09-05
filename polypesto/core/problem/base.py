from typing import Generic, List, Optional
from dataclasses import dataclass

import pandas as pd
from petab.v1 import Problem as PetabProblem

from polypesto.core.experiment import Experiment
from polypesto.core.petab import PetabData
from polypesto.models import ModelBase
from polypesto.core.problem import ProblemPaths, PypestoProblem
from polypesto.core.params import ParameterSet


@dataclass
class Problem:

    model: ModelBase
    petab_problem: PetabProblem
    pypesto_problem: PypestoProblem
    experiments: Optional[List[Experiment]] = None
    paths: Optional[ProblemPaths] = None
    id: Optional[str] = None

    @staticmethod
    def load(model: ModelBase, paths: ProblemPaths, **kwargs) -> "Problem":
        """
        Load a parameter estimation problem.

        Parameters
        ----------
        paths : ProblemPaths
            Defines locations of parameter estimation problem files.
        model : ModelInterface
            Model class to use for simulation.

        Returns
        -------
        Problem
            Loaded parameter estimation problem object
        """
        from polypesto.core.pypesto import load_pypesto_problem
        from polypesto.core.experiment import create_experiments_from_petab

        importer, problem = load_pypesto_problem(
            yaml_path=paths.petab_yaml, model_name=model.name, **kwargs
        )
        experiments = create_experiments_from_petab(importer.petab_problem)

        return Problem(
            model=model,
            petab_problem=importer.petab_problem,
            pypesto_problem=problem,
            experiments=experiments,
            paths=paths,
        )

    @staticmethod
    def from_experiments(
        data_dir: str,
        model: ModelBase,
        experiments: List[Experiment],
        problem_id: Optional[str] = None,
    ) -> "Problem":

        from polypesto.core.experiment import create_petab_from_experiments

        print("Creating problem from experiments...")
        print(f"Data directory: {data_dir}")
        paths = ProblemPaths(data_dir, problem_id)

        # Create PEtab problem from experiments
        cond_df, meas_df = create_petab_from_experiments(experiments)
        petab_data = PetabData(
            obs_df=model.get_obs_df(),
            cond_df=cond_df,
            param_df=model.get_param_df(),
            meas_df=meas_df,
            name=problem_id,
        )

        problem = write_and_load_problem(model, paths, petab_data)

        return problem

    def get_results(self):

        from pypesto import store

        return store.read_result(self.paths.pypesto_results)


def write_and_load_problem(
    model: ModelBase,
    paths: ProblemPaths,
    petab_data: PetabData,
    true_params: Optional[ParameterSet] = None,
) -> Problem:

    from polypesto.models import sbml
    from polypesto.core.petab import PetabIO

    sbml_filepath = sbml.write_model(
        model_def=model.sbml_model_def(), model_dir=paths.model
    )

    PetabIO.write_obs_df(petab_data.obs_df, filename=paths.observables)
    PetabIO.write_cond_df(petab_data.cond_df, filename=paths.conditions)
    PetabIO.write_param_df(petab_data.param_df, filename=paths.fit_parameters)
    PetabIO.write_meas_df(petab_data.meas_df, filename=paths.measurements)

    if true_params is not None:
        true_params.write(paths.true_params)

    print("Writing PEtab files...")
    PetabIO.write_yaml(
        yaml_filepath=paths.petab_yaml,
        sbml_filepath=sbml_filepath,
        cond_filepath=paths.conditions,
        meas_filepath=paths.measurements,
        obs_filepath=paths.observables,
        param_filepath=paths.fit_parameters,
    )

    return Problem.load(model, paths)
