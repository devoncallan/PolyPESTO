from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from polypesto.models import ModelBase
from polypesto.utils import write_json, read_json

from ..params import ParameterGroup, ParameterSet
from ..problem import (
    SimConditions,
    SimulatedProblem,
    ProblemPaths,
    run_parameter_estimation,
    simulate_problem,
)
from .core import (
    filter_study_dict,
    ResultsDict,
    EnsembleDict,
    SimulatedProblemDict,
    StudyKey,
    StudyMetadata,
    StudyPaths,
)


class Study:

    def __init__(
        self,
        model: ModelBase,
        true_params: ParameterGroup,
        metadata: StudyMetadata,
        problems: SimulatedProblemDict,
        paths: StudyPaths,
        keys: Optional[Sequence[StudyKey]] = None,
        results: Optional[ResultsDict] = None,
        ensembles: Optional[EnsembleDict] = None,
    ):
        self.model = model
        self.true_params = true_params
        self.problems = problems
        self.paths = paths
        self.metadata = metadata

        self.keys = keys
        self.results = results
        self.ensembles = ensembles

        self.name = paths.study_dir.stem

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
    def load(study_dir: str | Path, model: Optional[ModelBase] = None) -> Study:
        return load_study(study_dir, model)

    def get_conditions(self, prob_id: str) -> List[SimConditions]:
        prob_dict = self.get_problems(prob_id=prob_id)
        ref_prob = next(iter(prob_dict.values()))
        return ref_prob.sim_conditions

    def get_true_params(self, param_id: str) -> ParameterSet:
        if param_id not in self.true_params:
            raise KeyError(f"Parameter ID '{param_id}' not found in true parameters.")
        return self.true_params.get(param_id)

    def get_problems(
        self, prob_id: Optional[str] = None, param_id: Optional[str] = None
    ) -> SimulatedProblemDict:
        return filter_study_dict(self.problems, prob_id, param_id)

    def get_results(
        self, prob_id: Optional[str] = None, param_id: Optional[str] = None
    ) -> ResultsDict:
        if self.results is None:
            raise ValueError(
                "No results available. Please run parameter estimation first."
            )
        return filter_study_dict(self.results, prob_id, param_id)

    def get_ensembles(
        self, prob_id: Optional[str] = None, param_id: Optional[str] = None
    ) -> EnsembleDict:

        if self.ensembles is None:
            raise ValueError(
                "No ensembles available. Please run parameter estimation first."
            )
        return filter_study_dict(self.ensembles, prob_id, param_id)

    def run_parameter_estimation(
        self,
        config: Dict[str, Any],
        **kwargs,
    ) -> ResultsDict:
        """Run parameter estimation for all problems in the study."""

        num_problems = len(self.problems)
        for i, (key, problem) in enumerate(self.problems.items()):

            title_str = f"Problem {i + 1}/{num_problems}" + "-" * 50
            print("\n\n")
            print(title_str)
            print(f" | Problem ID: {key.prob_id}")
            print(
                f" | - Simulation Condition: {problem.sim_conditions[0].conds.to_dict()}"
            )
            print(f" | Parameter ID: {key.param_id}")
            print(f" | - True Parameters: {problem.true_params.to_dict()}")
            print(f" | Problem directory: {self.paths.prob_dir(key)}")
            print("-" * len(title_str))
            result = run_parameter_estimation(problem, config, **kwargs)
            self.results[key] = result

        return self.results

    def results_summary(self) -> pd.DataFrame:
        """
        Summarize results for all problems in the study.

        Returns a DataFrame with MultiIndex (problem_key, parameterId).
        """
        if self.results is None:
            raise ValueError(
                "No results available. Please run parameter estimation first."
            )

        summaries = {}
        for key, prob in self.problems.items():
            # key is already a StudyKey (string with format "prob_id | param_id")
            df = prob.results_summary()
            summaries[key] = df

        combined = pd.concat(summaries, names=["problem_key", "parameterId"])
        return combined


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
    all_sim_conds = {}
    keys = []

    prob_ids = list(sim_conds.keys())
    param_ids = true_params.get_ids()
    for param_id in param_ids:
        param_set = true_params[param_id]

        for prob_id in prob_ids:

            key = StudyKey(prob_id, param_id)
            keys.append(key)

            prob_dir = paths.prob_dir(key)

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
        keys=keys,
    )
    write_json(paths.metadata, metadata.to_dict())
    write_json(paths.true_params, true_params.to_dict())
    write_json(paths.model_config, model.to_config())

    return Study(model, true_params, metadata, problems, paths)


def load_study(study_dir: str | Path, model: Optional[ModelBase] = None) -> Study:

    paths = StudyPaths(study_dir)

    if not paths.study_dir.exists():
        raise FileNotFoundError(f"Study directory '{paths.study_dir}' does not exist.")

    metadata = StudyMetadata.load(paths.metadata)
    true_params = ParameterGroup.load(paths.true_params)

    # Load model from config if not provided
    if model is None:
        model_config = read_json(paths.model_config)
        model = ModelBase.from_config(model_config)

    problems: SimulatedProblemDict = {}
    results: ResultsDict = {}
    ensembles: EnsembleDict = {}

    for key in metadata.keys:

        prob = SimulatedProblem.load(paths.prob_dir(key), model)
        problems[key] = prob
        results[key] = prob.get_results()
        ensembles[key] = prob.get_ensemble()

    return Study(
        model,
        true_params,
        metadata,
        problems,
        paths,
        results=results,
        ensembles=ensembles,
    )
