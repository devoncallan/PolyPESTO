import os
from pathlib import Path


class ProblemPaths:
    """
    Manage file paths for a parameter estimation problem.

    Directory structure:

    `base_dir/petab/`
        - conditions.tsv
        - observables.tsv
        - parameters.tsv
        - measurements.tsv
        - petab.yaml
        - model.xml
        - params.json
    `base_dir/pypesto/`
        - results.hdf5
    `base_dir/figures/`
        - measurements.png
        - waterfall.png
        - profile.png
        - sampling_trace.png
        - ...
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.make_dirs()

    def make_dirs(self) -> None:

        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(self.petab_dir, exist_ok=True)
        os.makedirs(self.pypesto_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "ProblemPaths":
        yaml_path = Path(yaml_path)

        data_dir = yaml_path.parent.parent
        base_dir = data_dir.parent
        return ProblemPaths(base_dir)

    @property
    def _data_dir(self) -> str:
        return f"{self.base_dir}"

    ###################
    ### PEtab Files ###
    ###################

    @property
    def petab_dir(self) -> str:
        return f"{self._data_dir}/petab"

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
    def sbml_model(self) -> str:
        return f"{self.petab_dir}/sbml_model.xml"

    @property
    def petab_yaml(self) -> str:
        return f"{self.petab_dir}/petab.yaml"

    @property
    def measurements(self) -> str:
        return f"{self.petab_dir}/measurements.tsv"

    @property
    def true_params(self) -> str:
        return f"{self.petab_dir}/params.json"

    #####################
    ### PyPESTO Files ###
    #####################

    @property
    def pypesto_dir(self) -> str:
        return f"{self._data_dir}/pypesto"

    @property
    def pypesto_results(self) -> str:
        return f"{self.pypesto_dir}/results.hdf5"

    #####################
    ### Figures Files ###
    #####################

    @property
    def figures_dir(self) -> str:
        return f"{self._data_dir}/figures"

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

    #################
    ### Log Files ###
    #################

    @property
    def logs_dir(self) -> str:
        return f"{self._data_dir}/logs"

    @property
    def model_load_log(self) -> str:
        return f"{self.logs_dir}/model_load.log"
