from typing import Optional
from dataclasses import dataclass

import pandas as pd
from petab.v1 import Problem as PetabProblem

from polypesto.models import ModelInterface
from polypesto.core.problem import ProblemPaths, PypestoProblem


@dataclass
class Problem:
    """
    Represents experimental data for a
    single parameter estimation problem.
    """

    petab_problem: PetabProblem
    pypesto_problem: PypestoProblem
    paths: Optional[ProblemPaths] = None

    @staticmethod
    def load(paths: ProblemPaths, model: ModelInterface, **kwargs) -> "Problem":
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

        importer, problem = load_pypesto_problem(
            yaml_path=paths.petab_yaml, model_name=model.name, **kwargs
        )

        return Problem(
            petab_problem=importer.petab_problem,
            pypesto_problem=problem,
            paths=paths,
        )

    def get_conditions(self) -> pd.DataFrame:
        return self.petab_problem.condition_df
