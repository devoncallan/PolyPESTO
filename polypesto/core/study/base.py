from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from polypesto.core.conditions import SimConditions
from polypesto.core.problem.base import Problem
from polypesto.core.problem.simulate import simulate_problem
from polypesto.models.base import ModelBase
from polypesto.core import ParameterGroup

from .types import ConditionsDict, ProblemDict, ResultsDict
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
        results: Optional[ResultsDict] = None,
    ):
        self.model = model
        self.true_params = true_params
        self.sim_params = sim_params
        self.problems = problems
        self.results = results

    @staticmethod
    def create() -> Study:
        pass

    @staticmethod
    def load(study_dir: str | Path, model: ModelBase) -> Study:
        pass

    def run_parameter_estimation(
        self,
        config: Dict[str, Any],
        overwrite: bool = False,
    ) -> ResultsDict:
        pass


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

            key = (cond_id, param_id)
            key_str = f"{cond_id} | {param_id}"

            prob_dir = study_dir / param_id / cond_id

            problem_dirs[key_str] = str(prob_dir)

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


# def study_conditions_from_json(
#     sim_conds: Dict[str, Any], metadata: StudyMetadata
# ) -> ConditionsDict:

#     conds_dict: ConditionsDict = {}

#     param_ids = metadata.get_all_keys()
#     pass


def load_study(study_dir: str | Path, model: ModelBase) -> Study:

    paths = StudyPaths(study_dir)

    if not paths.study_dir.exists():
        raise FileNotFoundError(f"Study directory '{paths.study_dir}' does not exist.")

    metadata = read_json(paths.metadata)
    metadata = StudyMetadata.from_dict(metadata)

    true_params = read_json(paths.true_params)
    true_params = ParameterGroup.lazy_from_dict(true_params)

    sim_conds = read_json(paths.sim_params)
    # sim_conds = study_conditions_from_json(sim_conds)
    problems, results = load_data_from_metadata(model, metadata)

    return Study(model, true_params, sim_conds, problems, results)
