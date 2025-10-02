from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List


from polypesto.models import ModelBase
from polypesto.utils import write_json

from ..params import ParameterGroup
from ..problem import (
    SimulatedProblem,
    SimConditions,
    simulate_problem,
    run_parameter_estimation,
)

from .types import StudyKey, SimulatedProblemDict, ResultsDict
from .metadata import StudyMetadata
from .paths import StudyPaths


class Study:

    def __init__(
        self,
        model: ModelBase,
        true_params: ParameterGroup,
        problems: SimulatedProblemDict,
        paths: StudyPaths,
        results: ResultsDict = {},
    ):
        self.model = model
        self.true_params = true_params
        self.problems = problems
        self.paths = paths
        self.results = results

    @staticmethod
    def create(
        study_dir: str | Path,
        model: ModelBase,
        true_params: ParameterGroup,
        sim_conds: Dict[str, List[SimConditions]],
        **kwargs,
    ) -> Study:
        return create_study(study_dir, model, true_params, sim_conds, **kwargs)

    @staticmethod
    def load(study_dir: str | Path, model: ModelBase) -> Study:
        return load_study(study_dir, model)

    def run_parameter_estimation(
        self,
        config: Dict[str, Any],
        overwrite: bool = False,
    ) -> ResultsDict:
        """Run parameter estimation for all problems in the study."""

        for key, problem in self.problems.items():

            result = self.results.get(key, None)

            if overwrite or result is None:
                print(
                    f"Running parameter estimation for {key.param_id}, {key.param_id}..."
                )
                result = run_parameter_estimation(problem, config, result)
                self.results[key] = result
                print("Done.")
            else:
                print(f"Found existing result for {key.param_id}, {key.param_id}.")

        return self.results


def create_study(
    study_dir: str | Path,
    model: ModelBase,
    true_params: ParameterGroup,
    sim_conds: Dict[str, List[SimConditions]],
    **kwargs,
) -> Study:

    study_dir = Path(study_dir)
    paths = StudyPaths(study_dir)

    problems = {}
    problem_dirs = {}
    all_sim_conds = {}

    prob_ids = list(sim_conds.keys())
    param_ids = true_params.get_ids()
    for param_id in param_ids:
        param_set = true_params[param_id]

        for prob_id in prob_ids:

            key = StudyKey(prob_id, param_id)

            prob_dir = paths.prob_dir(key)
            problem_dirs[key] = str(prob_dir)

            sim_conds_list = sim_conds[prob_id]
            for sim_cond in sim_conds_list:
                sim_cond.true_params = param_set
            all_sim_conds[key] = sim_conds_list

            overwrite = kwargs.pop("overwrite", False)
            problem = simulate_problem(
                prob_dir, model, sim_conds_list, overwrite=overwrite
            )
            problems[key] = problem

    metadata = StudyMetadata(
        model_name=model.name,
        prob_ids=prob_ids,
        param_ids=param_ids,
        problem_dirs=problem_dirs,
    )
    write_json(paths.metadata, metadata.to_dict())
    write_json(paths.true_params, true_params.to_dict())

    return Study(model, true_params, problems, paths)


def load_study(study_dir: str | Path, model: ModelBase) -> Study:

    paths = StudyPaths(study_dir)

    if not paths.study_dir.exists():
        raise FileNotFoundError(f"Study directory '{paths.study_dir}' does not exist.")

    metadata = StudyMetadata.load(paths.metadata)
    true_params = ParameterGroup.load(paths.true_params)

    problems: SimulatedProblemDict = {}
    results: ResultsDict = {}
    for key, prob_dir in metadata.problem_dirs.items():

        problem = SimulatedProblem.load(prob_dir, model)
        problems[key] = problem
        results[key] = problem.get_results()

    return Study(model, true_params, problems, paths, results=results)
