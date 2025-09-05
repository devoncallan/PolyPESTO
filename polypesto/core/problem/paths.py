import os
from pathlib import Path
from typing import Optional


class ProblemPaths:

    def __init__(self, base_dir: str | Path, id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.id = id
        self.make_dirs()

    def make_dirs(self) -> None:

        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(self.petab_dir, exist_ok=True)
        os.makedirs(self.pypesto_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    @property
    def _data_dir(self) -> str:
        if self.id is None:
            return self.base_dir
        else:
            return f"{self.base_dir}/{self.id}"

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
