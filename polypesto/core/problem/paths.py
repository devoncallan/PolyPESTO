import os
from pathlib import Path
from typing import Optional


class ProblemPaths:

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.make_dirs()

    def make_dirs(self) -> None:

        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(self.petab_dir, exist_ok=True)
        os.makedirs(self.pypesto_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "ProblemPaths":
        yaml_path = Path(yaml_path)

        data_dir = yaml_path.parent.parent
        base_dir = data_dir.parent
        return ProblemPaths(base_dir)

    @property
    def _data_dir(self) -> str:
        return f"{self.base_dir}"

    @property
    def petab_dir(self) -> str:
        return f"{self._data_dir}/petab"

    @property
    def pypesto_dir(self) -> str:
        return f"{self._data_dir}/pypesto"

    @property
    def figures_dir(self) -> str:
        return f"{self._data_dir}/figures"

    @property
    def conditions(self) -> str:
        return f"{self.petab_dir}/conditions.tsv"

    @property
    def observables(self) -> str:
        return f"{self.petab_dir}/observables.tsv"

    @property
    def fit_parameters(self) -> str:
        return f"{self.petab_dir}/parameters.tsv"

    @property
    def model(self) -> str:
        return f"{self.petab_dir}/model.xml"

    @property
    def petab_yaml(self) -> str:
        return f"{self.petab_dir}/petab.yaml"

    @property
    def measurements(self) -> str:
        return f"{self.petab_dir}/measurements.tsv"

    @property
    def true_params(self) -> str:
        return f"{self.petab_dir}/params.json"

    @property
    def pypesto_results(self) -> str:
        return f"{self.pypesto_dir}/results.hdf5"

    @property
    def measurements_fig(self) -> str:
        return f"{self.figures_dir}/measurements.png"

    @property
    def waterfall_fig(self) -> str:
        return f"{self.figures_dir}/waterfall.png"

    @property
    def profile_fig(self) -> str:
        return f"{self.figures_dir}/profile.png"

    @property
    def sampling_trace_fig(self) -> str:
        return f"{self.figures_dir}/sampling_trace.png"

    @property
    def confidence_intervals_fig(self) -> str:
        return f"{self.figures_dir}/confidence_intervals.png"

    @property
    def sampling_scatter_fig(self) -> str:
        return f"{self.figures_dir}/sampling_scatter.png"

    @property
    def optimization_scatter_fig(self) -> str:
        return f"{self.figures_dir}/optimization_scatter.png"

    @property
    def ensemble_predictions_fig(self) -> str:
        return f"{self.figures_dir}/ensemble_predictions.png"

    @property
    def model_fit_fig(self) -> str:
        return f"{self.figures_dir}/model_fit.png"


def find_problem_dirs(base_dir: str | Path):

    base_dir = Path(base_dir)

    experiment_paths = {}
    for yaml_path in sorted(base_dir.glob("petab.yaml")):

        paths = ProblemPaths.from_yaml(yaml_path)

        # problem_id
        problem_id = yaml_path.parent.name
        experiment_paths[problem_id] = paths

    return experiment_paths


"""      
base_dir/
    problem_id/
        petab/
            conditions.tsv
            observables.tsv
            parameters.tsv
            measurements.tsv
            model.yaml
            model.xml
        pypesto/
            results.hdf5
        figures/
            measurements.png
            waterfall.png
            profile.png
            sampling_trace.png
            confidence_intervals.png
            sampling_scatter.png
            optimization_scatter.png
            ensemble_predictions.png
"""
