from __future__ import annotations
from typing import List, Dict
from pathlib import Path

import pandas as pd

from polypesto.core.params import ParameterGroup
from polypesto.core.problem import SimulatedProblem

from ..problem.core import ProblemFigure
from ..problem.simulate import SimConditions, check_sim_conditions_consistency
from .core import StudyKey, StudyPaths
from .study import Study


class StudyComparison(Dict[str, Study]):

    def __init__(self, studies: Dict[str, Study]):
        super().__init__(studies)
        self.ref_study_key = next(iter(self.keys()))
        self._validate()

    @classmethod
    def load(cls, study_dirs: List[str | Path], **kwargs) -> StudyComparison:

        studies = [Study.load(study_dir, **kwargs) for study_dir in study_dirs]
        return cls.from_studies(studies)

    @classmethod
    def from_studies(cls, studies: List[Study]) -> StudyComparison:
        return cls({study.name: study for study in studies})

    def _validate(self):
        if not self:
            raise ValueError("No studies to compare.")

        if len(self) < 2:
            raise ValueError("At least two studies are required for comparison.")

        ref_study = self[self.ref_study_key]
        ref_keys = ref_study.metadata.keys

        for name, study in self.items():

            if study.metadata.keys != ref_keys:
                print(f"Study '{name}' has inconsistent keys compared to the reference study.")
                

            for key in ref_keys:
                ref_problem = ref_study.problems[key]
                comp_problem = study.problems[key]

                if not check_sim_conditions_consistency(ref_problem.sim_conditions, comp_problem.sim_conditions):
                    print(f"Study '{name}' has inconsistent simulation conditions for problem '{key}'.")
                    # raise ValueError(
                    #     f"""
                    #     Study '{name}' has inconsistent simulation conditions for problem '{key}'.
                    #     Reference: {ref_problem.sim_conditions}
                    #     Comparison: {comp_problem.sim_conditions}
                    #     """
                    # )

                if ref_problem.true_params.to_dict() != comp_problem.true_params.to_dict():
                    print(f"Study '{name}' has inconsistent true parameters for problem '{key}'.")
                    # raise ValueError(
                    #     f"""
                    #     Study '{name}' has inconsistent true parameters for problem '{key}'.
                    #     Reference: {ref_problem.true_params}
                    #     Comparison: {comp_problem.true_params}
                    #     """
                    # )

    def get_keys(self) -> List[StudyKey]:
        return self[self.ref_study_key].metadata.keys

    def get_conds_dict(self) -> Dict[str, List[SimConditions]]:
        ref_study = self[self.ref_study_key]
        return {
            key.prob_id: ref_study.get_conditions(key.prob_id)
            for key in ref_study.metadata.keys
        }

    def get_true_params(self) -> ParameterGroup:
        ref_study = self[self.ref_study_key]
        return ref_study.true_params


    # def get_true_params(self) -> Dict[str, ParameterGroup]:
    #     ref_study = self[self.ref_study_key]
    #     return {
    #         key.param_id: ref_study.get_true_params(key.param_id)
    #         for key in ref_study.metadata.keys
    #     }

    def get_problems(self, key: StudyKey) -> Dict[str, SimulatedProblem]:

        problems = {}
        for name, study in self.items():
            problems[name] = study.problems[key]

        return problems

    def get_comparison(self, key: StudyKey) -> Dict[str, pd.DataFrame]:

        key_problems = self.get_problems(key)
        true_params = self[self.ref_study_key].get_true_params(key.param_id)

        comparison: Dict[str, pd.DataFrame] = {}
        # for name, val in true_params.items():
        for param_name in true_params.keys():

            dfs = []
            for study_name, problem in key_problems.items():

                df = problem.results_summary()
                dfs.append(df.loc[(param_name)])

            combined = pd.concat(dfs, keys=key_problems.keys(), names=["study"])
            comparison[param_name] = combined
        return comparison

    def get_figure_paths(
        self, key: StudyKey, fig_type: ProblemFigure
    ) -> Dict[str, Path]:

        figs = {}
        for name, study in self.items():
            problem = study.problems[key]

            fig_dir = problem.paths.figures_dir
            fig_path = fig_dir / fig_type.value

            if not fig_path.exists():
                raise FileNotFoundError(
                    f"Figure '{fig_type}' does not exist for study '{name}', problem '{key}' at path '{fig_path}'."
                )

            figs[name] = fig_path

        return figs


# class Study
