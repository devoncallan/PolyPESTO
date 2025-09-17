from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from ..petab import PetabData, PetabIO, PetabProblem
from ...models import sbml, ModelBase
from ..params import ParameterSet
from ..pypesto import PypestoProblem, load_pypesto_problem, set_solver_options
from ..experiment import Experiment, petab_to_experiments, experiments_to_petab
from ..problem import ProblemPaths
from ...utils.logging import redirect_output_to_file


@dataclass
class Problem:
    """A parameter estimation problem."""

    model: ModelBase
    petab_problem: PetabProblem
    pypesto_problem: PypestoProblem
    experiments: Optional[List[Experiment]] = None
    paths: Optional[ProblemPaths] = None
    id: Optional[str] = None

    @staticmethod
    def load(prob_dir: str | Path, model: ModelBase, **kwargs) -> "Problem":
        """
        Load a parameter estimation problem.

        Parameters
        ----------
        prob_dir : str | Path
            Directory containing the problem files.
        model : ModelInterface
            Model class to use for simulation.

        Returns
        -------
        Problem
            Loaded parameter estimation problem object
        """

        paths = ProblemPaths(prob_dir)

        with redirect_output_to_file(paths.model_load_log, mode="a"):
            model_name = model.model_name_with_hash()
            importer, pypesto_problem = load_pypesto_problem(
                yaml_path=paths.petab_yaml, model_name=model_name, **kwargs
            )
            pypesto_problem = set_solver_options(pypesto_problem, model.solver_options)

        experiments = petab_to_experiments(importer.petab_problem)

        return Problem(
            model=model,
            petab_problem=importer.petab_problem,
            pypesto_problem=pypesto_problem,
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
        print("Creating problem from experiments...")
        print(f"Data directory: {data_dir}")

        # Create PEtab problem from experiments
        cond_df, meas_df = experiments_to_petab(experiments)
        petab_data = PetabData(
            obs_df=model.get_obs_df(),
            cond_df=cond_df,
            param_df=model.get_param_df(),
            meas_df=meas_df,
            name=problem_id,
        )

        problem = write_petab(data_dir, model, petab_data)

        return problem

    def get_results(self):

        from pypesto import store

        try:
            return store.read_result(self.paths.pypesto_results)
        except:
            return None


def write_petab(
    data_dir: str | Path,
    model: ModelBase,
    petab_data: PetabData,
    true_params: Optional[ParameterSet] = None,
) -> Problem:
    """Write PEtab files to specified directory.

    Args:
        data_dir (str | Path): Directory to write PEtab files to.
        model (ModelBase): Model to use for simulation.
        petab_data (PetabData): PEtab data to write.
        true_params (Optional[ParameterSet]): True parameter values to write. Defaults to None.

    Returns:
        Problem: Created problem instance.
    """

    paths = ProblemPaths(data_dir)

    sbml_model = model.sbml_model
    sbml.write_model(sbml_model, paths.sbml_model)

    PetabIO.write_obs_df(petab_data.obs_df, filename=paths.observables)
    PetabIO.write_cond_df(petab_data.cond_df, filename=paths.conditions)
    PetabIO.write_param_df(petab_data.param_df, filename=paths.fit_parameters)
    PetabIO.write_meas_df(petab_data.meas_df, filename=paths.measurements)

    if true_params is not None:
        true_params.write(paths.true_params)

    print("Writing PEtab files...")
    PetabIO.write_yaml(
        yaml_filepath=paths.petab_yaml,
        sbml_filepath=paths.sbml_model,
        cond_filepath=paths.conditions,
        meas_filepath=paths.measurements,
        obs_filepath=paths.observables,
        param_filepath=paths.fit_parameters,
    )

    return Problem.load(data_dir, model)
