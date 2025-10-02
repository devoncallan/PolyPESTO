import os
from pathlib import Path

from polypesto.utils.file import filepath


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
        - sim_conds.json (optional)
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
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.petab_dir, exist_ok=True)

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "ProblemPaths":
        yaml_path = Path(yaml_path)

        data_dir = yaml_path.parent.parent
        base_dir = data_dir.parent
        return ProblemPaths(base_dir)

    ###################
    ### PEtab Files ###
    ###################

    @property
    def petab_dir(self) -> Path:
        return self.base_dir / "petab"

    @filepath
    def conditions(self) -> Path:
        return self.petab_dir / "conditions.tsv"

    @filepath
    def observables(self) -> Path:
        return self.petab_dir / "observables.tsv"

    @filepath
    def fit_parameters(self) -> Path:
        return self.petab_dir / "parameters.tsv"

    @filepath
    def sbml_model(self) -> Path:
        return self.petab_dir / "sbml_model.xml"

    @filepath
    def petab_yaml(self) -> Path:
        return self.petab_dir / "petab.yaml"

    @filepath
    def measurements(self) -> Path:
        return self.petab_dir / "measurements.tsv"

    @filepath
    def sim_conds(self) -> Path:
        return self.petab_dir / "sim_conds.json"

    #####################
    ### PyPESTO Files ###
    #####################

    @property
    def pypesto_dir(self) -> Path:
        return self.base_dir / "pypesto"

    @filepath
    def pypesto_results(self) -> Path:
        return self.pypesto_dir / "results.hdf5"

    ####################
    ### Ensemble Dir ###
    ####################

    @property
    def ensemble_dir(self) -> Path:
        return self.base_dir / "ensemble"

    #####################
    ### Figures Files ###
    #####################

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    @filepath
    def measurements_fig(self) -> Path:
        return self.figures_dir / "measurements.png"

    @filepath
    def waterfall_fig(self) -> Path:
        return self.figures_dir / "waterfall.png"

    @filepath
    def profile_fig(self) -> Path:
        return self.figures_dir / "profile.png"

    @filepath
    def sampling_trace_fig(self) -> Path:
        return self.figures_dir / "sampling_trace.png"

    @filepath
    def confidence_intervals_fig(self) -> Path:
        return self.figures_dir / "confidence_intervals.png"

    @filepath
    def sampling_scatter_fig(self) -> Path:
        return self.figures_dir / "sampling_scatter.png"

    @filepath
    def optimization_scatter_fig(self) -> Path:
        return self.figures_dir / "optimization_scatter.png"

    @filepath
    def ensemble_predictions_fig(self) -> Path:
        return self.figures_dir / "ensemble_predictions.png"

    @filepath
    def model_fit_fig(self) -> Path:
        return self.figures_dir / "model_fit.png"

    #################
    ### Log Files ###
    #################

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @filepath
    def model_load_log(self) -> Path:
        return self.logs_dir / "model_load.log"
