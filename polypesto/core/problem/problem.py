from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from polypesto.utils import redirect_output_to_file
from polypesto.models import ModelBase, sbml


from .. import petab as pet
from ..experiment import Experiment, experiments_to_petab, petab_to_experiments
from ..problem import ProblemPaths
from ..pypesto import (
    PypestoProblem,
    Result,
    Ensemble,
    EnsemblePrediction,
    create_ensemble,
    predict_with_ensemble,
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

        with redirect_output_to_file(paths.model_load_log, mode="a"):
            model_name = model.model_name_with_hash()
            importer, pypesto_problem = load_pypesto_problem(
                yaml_path=paths.petab_yaml, model_name=model_name, **kwargs
            )
            pypesto_problem = set_solver_options(pypesto_problem, model.solver_options)
            
            # print("MODEL PARAMS in Problem.load:")
            # print(pypesto_problem.objective.amici_model.getSolver().getSensitivityMethod())
            # print(pypesto_problem.objective.amici_model.getSolver().getSensitivityOrder())
            # print(pypesto_problem.objective.amici_model.getSolver().getReturnDataReportingMode())
            # print("==========")
            
            # print("MODEL PARAMS from MODEL in Problem.load:")
            # print(pypesto_problem.objective.amici_solver.getSensitivityMethod())
            # print(pypesto_problem.objective.amici_solver.getSensitivityOrder())
            # print(pypesto_problem.objective.amici_solver.getReturnDataReportingMode())
            # print("==========")

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
    ) -> Problem:
        """Create a parameter estimation problem from experiments."""

        obs_df = model.get_obs_df()
        param_df = model.get_param_df()
        cond_df, meas_df = experiments_to_petab(experiments, model.obs_noise_map)
        petab_data = pet.PetabData(obs_df, cond_df, param_df, meas_df, problem_id)
        petab_data.write(output_dir, model.sbml_model)

        return Problem.load(output_dir, model)

    def get_results(self) -> Result | None:

        return load_result(self.paths.pypesto_results)

    def visualize_results(self, **kwargs) -> None:
        from polypesto.vis import plot_results

        result = self.get_results()
        if result is None:
            return
        plot_results(result, self, **kwargs)

    def run_parameter_estimation(
        self, config: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Optional[Result]:
        return run_parameter_estimation(self, config, **kwargs)

    def ensemble_prediction(
        self, ensemble_prob: Problem, **kwargs
    ) -> Tuple[Ensemble, EnsemblePrediction] | None:
        return ensemble_prediction(self, ensemble_prob, **kwargs)


def run_parameter_estimation(
    prob: Problem,
    config: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    save: bool = True,
    plot: bool = True,
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
            # print(f"\tUsing existing `{key}` results - skipping")
            return _result

        problem = prob.pypesto_problem
        kwargs = dict(result=_result, **config[key])
        if key == "optimize":
            return optimize_problem(problem=problem, **kwargs)
        elif key == "profile":
            return profile_problem(problem=problem, **kwargs)
        elif key == "sample":
            return sample_problem(problem=problem, **kwargs)
        else:
            raise ValueError(f"Unknown key: {key}")

    result = deepcopy(existing_result)
    result = run_if_found("optimize", result)
    result = run_if_found("profile", result)
    result = run_if_found("sample", result)

    if result and save:
        save_result(
            result, prob.paths.pypesto_results, overwrite=overwrite, **save_components
        )

    if result and plot:

        prob.visualize_results(overwrite=overwrite)

    return result


def ensemble_prediction(
    prob: Problem, ensemble_prob: Problem, plot: bool = True
) -> Tuple[Ensemble, EnsemblePrediction] | None:

    result = prob.get_results()
    if result is None:
        print("No results found for problem - cannot create ensemble predictions")
        return None

    ensemble = create_ensemble(prob.pypesto_problem, result)
    ensemble_pred = predict_with_ensemble(
        ensemble, ensemble_prob.pypesto_problem, output_type="y"
    )

    if plot:
        from polypesto.vis import plot_ensemble_predictions, save_plot

        with save_plot(prob.paths.ensemble_predictions_fig):
            plot_ensemble_predictions(ensemble_pred, prob)

    return ensemble, ensemble_pred
