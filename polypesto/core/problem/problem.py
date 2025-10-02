from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from polypesto.utils import redirect_output_to_file

from ...models import ModelBase, sbml
from .. import petab as pet
from ..experiment import Experiment, experiments_to_petab, petab_to_experiments
from ..problem import ProblemPaths
from ..pypesto import (
    PypestoProblem,
    Result,
    has_results,
    load_pypesto_problem,
    load_result,
    optimize_problem,
    profile_problem,
    sample_problem,
    save_result,
    set_solver_options,
)


@dataclass
class Problem:
    """A parameter estimation problem."""

    model: ModelBase
    petab_problem: pet.PetabProblem
    pypesto_problem: PypestoProblem
    paths: ProblemPaths
    experiments: List[Experiment]

    @staticmethod
    def load(prob_dir: str | Path, model: ModelBase, **kwargs) -> Problem:
        """
        Load a parameter estimation problem.

        Parameters
        ----------
        prob_dir : str | Path
            Directory containing the problem files.
        model : ModelBase
            Model class to use for simulation.

        Returns
        -------
        Problem
            Loaded parameter estimation problem object
        """

        paths = ProblemPaths(prob_dir)

        msg = f"Loading problem from {prob_dir}"
        with redirect_output_to_file(paths.model_load_log, mode="a", message=msg):
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
        output_dir: str,
        model: ModelBase,
        experiments: List[Experiment],
        problem_id: Optional[str] = None,
    ) -> "Problem":
        print("Creating problem from experiments...")
        print(f"Output directory: {output_dir}")

        # Create PEtab problem from experiments
        cond_df, meas_df = experiments_to_petab(experiments)
        petab_data = pet.PetabData(
            obs_df=model.get_obs_df(),
            cond_df=cond_df,
            param_df=model.get_param_df(),
            meas_df=meas_df,
            name=problem_id,
        )

        problem = write_petab(output_dir, model, petab_data)

        return problem

    def get_results(self) -> Optional[Result]:

        return load_result(self.paths.pypesto_results)


def write_petab(
    data_dir: str | Path,
    model: ModelBase,
    petab_data: pet.PetabData,
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

    pet.write_observable_df(petab_data.obs_df, paths.observables)
    pet.write_condition_df(petab_data.cond_df, paths.conditions)
    pet.write_parameter_df(petab_data.param_df, paths.fit_parameters)
    pet.write_measurement_df(petab_data.meas_df, paths.measurements)

    print("Writing PEtab files...")
    pet.PetabIO.write_yaml(
        yaml_filepath=paths.petab_yaml,
        sbml_filepath=paths.sbml_model,
        cond_filepath=paths.conditions,
        meas_filepath=paths.measurements,
        obs_filepath=paths.observables,
        param_filepath=paths.fit_parameters,
    )

    return Problem.load(data_dir, model)


def run_parameter_estimation(
    prob: Problem,
    config: Optional[Dict[str, Any]] = None,
    save: bool = True,
    overwrite: bool = True,
) -> Result:

    if config is None or len(config) == 0:
        print("No parameter estimation steps configured - skipping")
        return None

    save_components: Dict[str, bool] = {"problem": True}
    save_components.update({key: True for key in config.keys()})

    existing_result = prob.get_results()

    def run_if_found(key: str, _result: Optional[Result] = None) -> Optional[Result]:

        if key not in config:
            return _result

        if not overwrite and has_results(existing_result, key):
            print(f"\tUsing existing `{key}` results - skipping")
            return _result

        kwargs = dict(problem=prob.pypesto_problem, **config[key])
        if key == "optimize":
            print("\tRunning optimize_problem...")
            return optimize_problem(result=_result, **kwargs)
        elif key == "profile":
            print("\tRunning profile_problem...")
            return profile_problem(result=_result, **kwargs)
        elif key == "sample":
            print("\tRunning sample_problem...")
            return sample_problem(result=_result, **kwargs)
        else:
            raise ValueError(f"Unknown key: {key}")

    result = deepcopy(existing_result)
    result = run_if_found("optimize", result)
    result = run_if_found("profile", result)
    result = run_if_found("sample", result)

    print(result)
    if result and save:
        print(f"\tSaving results to {prob.paths.pypesto_results}")

        try:
            save_result(
                result=result,
                filename=prob.paths.pypesto_results,
                overwrite=overwrite,
                **save_components,
            )
        except RuntimeError:
            if overwrite:
                print("Error saving results despite overwrite=True")

    return result
