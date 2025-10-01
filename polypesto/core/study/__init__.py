from pathlib import Path
from typing import TypeAlias, Dict, List, Optional, Tuple, Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from ..models import ModelBase
from .problem import Problem, ProblemPaths, simulate_problem
from .pypesto import Result
from .conditions import SimConditions, create_sim_conditions
from .params import ParameterGroup, ParameterSet
from ..utils.file import read_json, write_json
from pypesto import store

ProbParamKey: TypeAlias = Tuple[str, str]
T = TypeVar("T")
ProbParamDict: TypeAlias = Dict[ProbParamKey, T]

ProblemDict: TypeAlias = ProbParamDict[Problem]
ResultsDict: TypeAlias = ProbParamDict[Result]
ConditionsDict: TypeAlias = ProbParamDict[List[SimConditions]]
PathsDict: TypeAlias = ProbParamDict[ProblemPaths]


class Study:

    def __init__(
        self,
        model: ModelBase,
        true_params: ParameterGroup,
        sim_params: ConditionsDict,
        problems: ProblemDict,
        results: ResultsDict,
    ):
        self.model = model
        self.true_params = true_params
        self.sim_params = sim_params
        self.problems = problems
        self.results = results

        self._param_ids = sorted(
            list(set(param_id for _, param_id in self.problems.keys()))
        )
        self._prob_ids = sorted(
            list(set(prob_id for prob_id, _ in self.problems.keys()))
        )

    @staticmethod
    def create(
        study_dir: str,
        model: ModelBase,
        true_params: ParameterGroup,
        sim_conds: ConditionsDict,
        overwrite: bool = False,
    ) -> "Study":

        if not overwrite:
            try:
                study = Study.load(study_dir, model)
                print(
                    f"Found existing study in {study_dir}. Set overwrite=True to recreate."
                )
                return study
            except FileNotFoundError:
                pass
        return create_study(study_dir, model, true_params, sim_conds)

    @staticmethod
    def load(study_dir: str | Path, model: ModelBase) -> "Study":
        return load_study(study_dir, model)

    def get_prob_ids(self) -> List[str]:
        """Get all unique problem IDs in the study."""
        return self._prob_ids

    def get_param_ids(self) -> List[str]:
        """Get all unique parameter IDs in the study."""
        return self._param_ids

    def get_problems(
        self,
        filter_prob_id: Optional[str] = None,
        filter_param_id: Optional[str] = None,
    ) -> ProblemDict:
        """Get filtered problems."""
        return _filter_dict(self.problems, filter_prob_id, filter_param_id)

    def get_results(
        self,
        filter_prob_id: Optional[str] = None,
        filter_param_id: Optional[str] = None,
    ) -> ResultsDict:
        """Get filtered results."""
        return _filter_dict(self.results, filter_prob_id, filter_param_id)

    def run_parameter_estimation(
        self,
        config: Dict[str, Any],
        overwrite: bool = False,
    ) -> ResultsDict:
        """Run parameter estimation for all problems in the study."""
        from .problem.estimate import run_parameter_estimation

        for (prob_id, param_id), problem in self.problems.items():
            key = (prob_id, param_id)

            result = self.results.get(key, None)

            if overwrite or result is None:
                print(f"Running parameter estimation for {prob_id}, {param_id}...")
                result = run_parameter_estimation(problem, config, result)
                self.results[key] = result
                print("Done.")
            else:
                print(f"Found existing result for {prob_id}, {param_id}.")

        return self.results


def _filter_dict(
    prob_param_dict: ProbParamDict[T],
    filter_prob_id: Optional[str],
    filter_param_id: Optional[str],
) -> ProbParamDict[T]:
    """Filter a problem-parameter dictionary by prob_id and/or param_id."""
    filtered_dict = {}
    for (prob_id, param_id), value in prob_param_dict.items():
        if (filter_prob_id is None or prob_id == filter_prob_id) and (
            filter_param_id is None or param_id == filter_param_id
        ):
            filtered_dict[(prob_id, param_id)] = value
    return filtered_dict


def _filter_by_conditions(filter_conds: Dict[str, float]):
    
    
    
    
    pass


"""
# Inner list: correspond to conditions for multiple experiments
# Outer list: correspond to experiments for a single problem
conds = {
    "A0": [[0.25, 0.50], [0.50, 0.75], [0.25, 0.75]],
    "B0": [[0.75, 0.50], [0.50, 0.25], [0.75, 0.25]],
}

conds_list = [
    {"A0": [0.25, 0.50], "B0": [0.75, 0.50]},
    {"A0": [0.50, 0.75], "B0": [0.50, 0.25]},
    {"A0": [0.25, 0.75], "B0": [0.75, 0.25]},
]

t_evals: ArrayLike = np.linspace(0, 1, 100)



"""


def _transpose_study_conditions(
    conds: Dict[str, List[ArrayLike]],
) -> List[Dict[str, np.ndarray]]:
    """Transpose study conditions from `dict of lists` to `list of dicts`.

    conds = {
        "A0": [[0.25, 0.50], [0.50, 0.75], [0.25, 0.75]],
        "B0": [[0.75, 0.50], [0.50, 0.25], [0.75, 0.25]],
    }

    conds_list = [
        {"A0": [0.25, 0.50], "B0": [0.75, 0.50]},
        {"A0": [0.50, 0.75], "B0": [0.50, 0.25]},
        {"A0": [0.25, 0.75], "B0": [0.75, 0.25]},
    ]

    """

    if not conds:
        return []

    conds_list: List[Dict[str, ArrayLike]] = []

    num_exps_dict = {k: len(v) for k, v in conds.items()}
    num_exps_list = list(num_exps_dict.values())
    assert all(
        n == num_exps_list[0] for n in num_exps_list
    ), f"All condition lists must have the same length. Actual lengths: {num_exps_dict}"

    num_exps = num_exps_list[0]
    for i in range(num_exps):
        exp_conds = {k: np.array(v[i]) for k, v in conds.items()}

        assert (
            len(set(len(cond) for cond in exp_conds.values())) == 1
        ), f"All conditions in a single experiment {i} must have the same length. Actual lengths: {[len(cond) for cond in exp_conds.values()]}"

        conds_list.append(exp_conds)

    return conds_list


def create_study_conditions(
    conds: Dict[str, List[ArrayLike]],
    t_evals: ArrayLike | List[ArrayLike],
    noise_levels: float | List[float] = 0.0,
) -> Dict[str, List[SimConditions]]:

    sim_conds: Dict[str, List[SimConditions]] = {}

    conds_list = _transpose_study_conditions(conds)
    prob_ids = [f"prob_{i}" for i in range(len(conds_list))]
    raw_conds_dict = dict(zip(prob_ids, conds_list))

    # Create simulation conditions for each parameter set
    for prob_id, raw_conds in raw_conds_dict.items():

        sim_conds[prob_id] = create_sim_conditions(
            true_params=ParameterSet.empty(),
            conds=raw_conds,
            t_evals=t_evals,
            noise_levels=noise_levels,
        )

    return sim_conds


def create_study(
    study_dir: str,
    model: ModelBase,
    true_params: ParameterGroup,
    sim_conds: Dict[str, List[SimConditions]],
) -> Study:

    problems = {}
    metadata = {
        "model_name": model.name,
        "problem_dirs": {},
        "param_ids": true_params.get_ids(),
        "cond_ids": list(sim_conds.keys()),
    }

    conds_dict = {k: [cond.to_dict() for cond in v] for k, v in sim_conds.items()}

    for param_id in metadata["param_ids"]:
        param_set = true_params.by_id(param_id)

        for prob_id in metadata["cond_ids"]:
            key = (prob_id, param_id)
            key_str = f"{prob_id} | {param_id}"

            prob_dir = Path(study_dir) / f"{param_id}" / f"{prob_id}"
            metadata["problem_dirs"][key_str] = str(prob_dir)

            sim_conds_list = sim_conds[prob_id]
            for sim_cond in sim_conds_list:
                sim_cond.true_params = param_set

            problem = simulate_problem(prob_dir, model, sim_conds_list)
            problems[key] = problem

    write_json(f"{study_dir}/metadata.json", metadata)
    write_json(f"{study_dir}/true_params.json", true_params.to_dict())
    write_json(f"{study_dir}/sim_conds.json", conds_dict)

    return Study(model, true_params, sim_conds, problems, {})


########################
# LOAD STUDY FUNCTIONS #
########################


def study_conditions_from_json(
    sim_conds: Dict[str, Any],
    true_params: ParameterGroup,
) -> ConditionsDict:
    """Convert study conditions from JSON format to internal representation."""

    conditions_dict: ConditionsDict = {}
    param_ids = true_params.get_ids()
    for param_id in param_ids:
        param_set = true_params.by_id(param_id)
        for prob_id, cond_list in sim_conds.items():
            key = (prob_id, param_id)
            conditions_dict[key] = []
            for i in range(len(cond_list)):
                cond_list[i]["true_params"] = param_set.to_dict()
                conditions_dict[key].append(SimConditions.from_dict(cond_list[i]))

    return conditions_dict


def load_data_from_metadata(
    model: ModelBase,
    metadata: Dict[str, Any],
) -> Tuple[ProblemDict, ResultsDict]:

    problems = {}
    results = {}
    for key_str, prob_dir in metadata["problem_dirs"].items():
        prob_id, param_id = key_str.split(" | ")
        key = (prob_id, param_id)

        paths = ProblemPaths(prob_dir)
        problems[key] = Problem.load(model=model, paths=paths)

        if Path(paths.pypesto_results).exists():
            results[key] = store.read_result(paths.pypesto_results)

    return problems, results


def load_study(study_dir: str | Path, model: ModelBase) -> Study:

    study_dir = Path(study_dir)
    if not study_dir.exists():
        raise ValueError(f"Study directory {study_dir} does not exist.")

    try:
        metadata = read_json(study_dir / "metadata.json")
        true_params = read_json(study_dir / "true_params.json")
        sim_conds = read_json(study_dir / "sim_conds.json")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find metadata files in {study_dir}.") from e

    required_keys = ["model_name", "problem_dirs", "param_ids", "cond_ids"]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Metadata is missing required key: {key}")
    assert (
        metadata["model_name"] == model.name
    ), f"Model name in metadata ({metadata['model_name']}) does not match provided model name ({model.name})."

    true_params = ParameterGroup.lazy_from_dict(true_params)
    sim_params = study_conditions_from_json(sim_conds, true_params)
    problems, results = load_data_from_metadata(model, metadata)

    return Study(
        model=model,
        true_params=true_params,
        sim_params=sim_params,
        problems=problems,
        results=results,
    )
