from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from polypesto.core.conditions import SimConditions
from polypesto.core.problem.base import Problem
from polypesto.core.problem.simulate import simulate_problem
from polypesto.models.base import ModelBase
from polypesto.core import ParameterGroup

from .types import ConditionsDict, ProblemDict, ResultsDict, StudyKey
from .metadata import StudyMetadata
from ...utils.file import read_json, write_json

from .paths import StudyPaths


class Study:

    def __init__(
        self,
        model: ModelBase,
        true_params: ParameterGroup,
        sim_params: ConditionsDict,
        problems: ProblemDict,
        results: ResultsDict = {},
    ):
        self.model = model
        self.true_params = true_params
        self.sim_params = sim_params
        self.problems = problems
        self.results = results

    @staticmethod
    def create(
        study_dir: str | Path,
        model: ModelBase,
        true_params: ParameterGroup,
        sim_conds: Dict[str, List[SimConditions]],
    ) -> Study:
        return create_study(study_dir, model, true_params, sim_conds)

    @staticmethod
    def load(study_dir: str | Path, model: ModelBase) -> Study:
        return load_study(study_dir, model)

    def run_parameter_estimation(
        self,
        config: Dict[str, Any],
        overwrite: bool = False,
    ) -> ResultsDict:
        """Run parameter estimation for all problems in the study."""
        from ..problem.estimate import run_parameter_estimation

        for key, problem in self.problems.items():

            result = self.results.get(key, None)

            if overwrite or result is None:
                print(
                    f"Running parameter estimation for {key.cond_id}, {key.param_id}..."
                )
                result = run_parameter_estimation(problem, config, result)
                self.results[key] = result
                print("Done.")
            else:
                print(f"Found existing result for {key.cond_id}, {key.param_id}.")

        return self.results


def create_study(
    study_dir: str | Path,
    model: ModelBase,
    true_params: ParameterGroup,
    sim_conds: Dict[str, List[SimConditions]],
) -> Study:

    study_dir = Path(study_dir)
    param_ids = true_params.get_ids()
    cond_ids = list(sim_conds.keys())
    conds_dict = {k: [cond.to_dict() for cond in v] for k, v in sim_conds.items()}

    problems = {}
    problem_dirs = {}
    for param_id in param_ids:
        param_set = true_params.by_id(param_id)

        for cond_id in cond_ids:
            
            key = StudyKey(cond_id, param_id)

            prob_dir = study_dir / param_id / cond_id

            problem_dirs[key] = str(prob_dir)

            sim_conds_list = sim_conds[cond_id]
            for sim_cond in sim_conds_list:
                sim_cond.true_params = param_set

            problem = simulate_problem(prob_dir, model, sim_conds_list)
            problems[key] = problem

    metadata = StudyMetadata(
        model_name=model.name,
        param_ids=param_ids,
        cond_ids=cond_ids,
        problem_dirs=problem_dirs,
    )
    write_json(study_dir / "metadata.json", metadata.to_dict())
    write_json(study_dir / "true_params.json", true_params.to_dict())
    write_json(study_dir / "sim_conds.json", conds_dict)

    return Study(model, true_params, sim_conds, problems)


def load_data_from_metadata(
    model: ModelBase, metadata: StudyMetadata
) -> Tuple[ProblemDict, ResultsDict]:

    problems: ProblemDict = {}
    results: ResultsDict = {}
    for key, prob_dir in metadata.problem_dirs.items():

        problem = Problem.load(prob_dir, model)
        problems[key] = problem
        results[key] = problem.get_results()

    return problems, results


def study_conditions_from_json(
    sim_conds: Dict[str, Any], true_params: ParameterGroup
) -> ConditionsDict:

    conds_dict: ConditionsDict = {}

    param_ids = true_params.get_ids()
    for param_id in param_ids:
        param_set = true_params.by_id(param_id)

        for prob_id, cond_list in sim_conds.items():
            key = StudyKey(prob_id, param_id)
            conds_dict[key] = []
            for i in range(len(cond_list)):
                cond_list[i]["true_params"] = param_set.to_dict()
                conds_dict[key].append(SimConditions.from_dict(cond_list[i]))

    return conds_dict


def load_study(study_dir: str | Path, model: ModelBase) -> Study:

    paths = StudyPaths(study_dir)

    if not paths.study_dir.exists():
        raise FileNotFoundError(f"Study directory '{paths.study_dir}' does not exist.")

    raw_metadata = read_json(paths.metadata)
    metadata = StudyMetadata.from_dict(raw_metadata)

    raw_true_params = read_json(paths.true_params)
    true_params = ParameterGroup.lazy_from_dict(raw_true_params)

    raw_sim_conds = read_json(paths.sim_params)
    sim_conds = study_conditions_from_json(raw_sim_conds, true_params)

    problems, results = load_data_from_metadata(model, metadata)

    return Study(model, true_params, sim_conds, problems, results)
