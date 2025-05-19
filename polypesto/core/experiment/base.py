from typing import Optional
from dataclasses import dataclass

import pandas as pd
from pypesto.problem import Problem as PypestoProblem
from petab.v1 import Problem as PetabProblem

from polypesto.models import ModelInterface

from polypesto.core.experiment import ExperimentPaths


@dataclass
class Experiment:
    """
    Represents experimental data for a
    single parameter estimation problem.
    """

    petab_problem: PetabProblem
    pypesto_problem: PypestoProblem
    paths: Optional[ExperimentPaths] = None

    @staticmethod
    def load(paths: ExperimentPaths, model: ModelInterface, **kwargs) -> "Experiment":
        """
        Load an experiment.

        Parameters
        ----------
        paths : ExperimentPaths
            Paths object containing locations of experiment files
        model : ModelInterface
            Model class to use for simulation

        Returns
        -------
        Experiment
            Loaded experiment object
        """
        from polypesto.core.pypesto import load_pypesto_problem

        importer, problem = load_pypesto_problem(
            yaml_path=paths.petab_yaml, model_name=model.name, **kwargs
        )

        return Experiment(
            petab_problem=importer.petab_problem,
            pypesto_problem=problem,
            paths=paths,
        )
        
    def get_conditions(self) -> pd.DataFrame:
        return self.petab_problem.condition_df