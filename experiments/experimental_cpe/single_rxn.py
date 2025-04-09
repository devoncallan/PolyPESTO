import numpy as np
import matplotlib.pyplot as plt

from polypesto.core.experiment import Experiment
from polypesto.core.experiment.paths import ExperimentPaths
from polypesto.models.CRP2 import IrreversibleCPE
from polypesto.core.experiment import run_parameter_estimation
from polypesto.core.pypesto import (
    create_ensemble,
    predict_with_ensemble,
)
from polypesto.visualization import (
    plot_optimization_scatter,
    plot_sampling_scatter,
    plot_confidence_intervals,
    plot_waterfall,
    plot_parameter_traces,
    plot_profiles,
    plot_ensemble_predictions,
    plot_all_measurements,
)

# define where your data lives
EXPERIMENT_BASE_DIR = "/PolyPESTO/experiments/experimental_cpe"
EXPERIMENT_ID = "p_000"

# construct paths and load experiment
exp_paths = ExperimentPaths(base_dir=EXPERIMENT_BASE_DIR, exp_id=EXPERIMENT_ID)
exp = Experiment.load(paths=exp_paths, model=IrreversibleCPE)

# Run parameter estimation
result = run_parameter_estimation(
    exp,
    config=dict(
        optimize=dict(n_starts=100, method="Nelder-Mead"),
        profile=dict(method="Nelder-Mead"),
        sample=dict(n_samples=10000, n_chains=5),
    ),
)

quit()

# for (cond_id, p_id), result in study.results.items():

#     exp = study.experiments[(cond_id, p_id)]
#     true_params = exp.true_params

#     print(f"Experiment: {cond_id}, Parameter set: {p_id}")

#     ensemble = create_ensemble(exp, result)

#     test_exp = list(test_study.experiments.values())[0]
#     ensemble_pred = predict_with_ensemble(ensemble, test_exp, output_type="y")

#     fig, axs = plot_ensemble_predictions(ensemble_pred, test_exp)
#     fig.savefig(exp.paths.ensemble_predictions_plot, dpi=300)

#     axs = plot_all_measurements(exp.petab_problem.measurement_df)
#     fig = plt.gcf()
#     fig.savefig(exp.paths.measurements_data_plot, dpi=300)

#     fig, ax = plot_optimization_scatter(result, true_params.to_dict())
#     fig.savefig(exp.paths.optimization_scatter_plot, dpi=300)

#     fig, ax = plot_waterfall(result)
#     fig.savefig(exp.paths.waterfall_plot, dpi=300)

#     fig, ax = plot_sampling_scatter(result, true_params.to_dict())
#     fig.savefig(exp.paths.sampling_scatter_plot, dpi=300)

#     fig, ax = plot_parameter_traces(result, true_params.to_dict())
#     fig.savefig(exp.paths.sampling_trace_plot, dpi=300)

#     fig, ax = plot_profiles(result, true_params.to_dict())
#     fig.savefig(exp.paths.profile_plot, dpi=300)

#     fig, ax = plot_confidence_intervals(result, true_params.to_dict())
#     fig.savefig(exp.paths.confidence_intervals_plot, dpi=300)

#     plt.close("all")
