import numpy as np
import matplotlib.pyplot as plt

import pypesto
import pypesto.sample

from polypesto.core.params import ParameterGroup
from polypesto.core.study import Study, create_study
from polypesto.core.experiment import create_simulation_conditions
from polypesto.models.CRP2 import IrreversibleCPE
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
    plot_all_comparisons,
)
from polypesto.utils.paths import setup_data_dirs

DATA_DIR, TEST_DIR = setup_data_dirs(__file__)

# simulation_params = ParameterGroup.create_parameter_grid(
#     {
#         "rA": [0.1, 0.5, 1.0, 2.0, 10.0],
#         "rB": [0.1, 0.5, 1.0, 2.0, 10.0],
#     }
# )

# # Define fitting parameters
# fit_params = IrreversibleCPE.get_default_parameters()

# # Define experimental configurations
# ntrials = 1
# t_eval = np.arange(0, 1, 0.1)

# fA0s = [[0.03, 0.05, 0.1]]
# cM0s = [[1.0, 1.0, 1.0]]
# names = [f"fA0_{fA0}_cM0_{cM0}" for fA0, cM0 in zip(fA0s, cM0s)]
# n_trials = len(fA0s)
# assert len(fA0s) == len(cM0s), "fA0s and cM0s must have the same length"

# conditions = create_simulation_conditions(
#     dict(
#         name=names,
#         t_eval=[t_eval] * ntrials,
#         conditions=dict(fA0=fA0s, cM0=cM0s),
#         fit_params=[fit_params] * ntrials,
#         noise_level=[0.02] * ntrials,
#     )
# )

# # Create the study - this will simulate all experiments
# study = create_study(
#     model=IrreversibleCPE,
#     simulation_params=simulation_params,
#     conditions=conditions,
#     base_dir=DATA_DIR,
#     overwrite=True,
# )
study = Study.load(DATA_DIR, IrreversibleCPE)

# # Create testing conditions for ensemble predictions
# fA0s_test = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
# cM0s_test = [[1.0] * 9]
# test_conditions = create_simulation_conditions(
#     dict(
#         name=["Test_Study"],
#         t_eval=[t_eval],
#         conditions=dict(fA0=fA0s_test, cM0=cM0s_test),
#         fit_params=[fit_params],
#         noise_level=[0.02],
#     )
# )
# test_study = create_study(
#     model=IrreversibleCPE,
#     simulation_params=simulation_params,
#     conditions=test_conditions,
#     base_dir=TEST_DIR,
# )

# test_study = Study.load(TEST_DIR, IrreversibleCPE)

# # Run parameter estimation
# study.run_parameter_estimation(
#     config=dict(
#         optimize=dict(n_starts=100, method="Nelder-Mead"),
#         profile=dict(method="Nelder-Mead"),
#         sample=dict(n_samples=10000, n_chains=5),
#     )
# )

# for (cond_id, p_id), result in study.results.items():

plot_all_comparisons(study)

# print(result.sample_result)

# # for ci in pypesto.sample.calculate_ci_mcmc_sample(result):
# #     print((10 ** ci[0], 10 ** ci[1]))

# lbs, ubs = pypesto.sample.calculate_ci_mcmc_sample(result)
# print(lbs, ubs)

# print([(10**lb, 10**ub) for (lb, ub) in zip(lbs, ubs)])

# # print(
# #     [
# #         (10 ** ci[0], 10 ** ci[1])
# #         for ci in pypesto.sample.calculate_ci_mcmc_sample(result)
# #     ]
# # )

# exp = study.experiments[(cond_id, p_id)]
# true_params = exp.true_params

# print(f"Experiment: {cond_id}, Parameter set: {p_id}")
# break

# ensemble = create_ensemble(exp, result)

# test_exp = list(test_study.experiments.values())[0]
# ensemble_pred = predict_with_ensemble(ensemble, test_exp, output_type="y")

# fig, axs = plot_ensemble_predictions(ensemble_pred, test_exp)
# fig.savefig(exp.paths.ensemble_predictions_plot, dpi=300)

# axs = plot_all_measurements(exp.petab_problem.measurement_df)
# fig = plt.gcf()
# fig.savefig(exp.paths.measurements_data_plot, dpi=300)

# fig, ax = plot_optimization_scatter(result, true_params.to_dict())
# fig.savefig(exp.paths.optimization_scatter_plot, dpi=300)

# fig, ax = plot_waterfall(result)
# fig.savefig(exp.paths.waterfall_plot, dpi=300)

# fig, ax = plot_sampling_scatter(result, true_params.to_dict())
# fig.savefig(exp.paths.sampling_scatter_plot, dpi=300)

# fig, ax = plot_parameter_traces(result, true_params.to_dict())
# fig.savefig(exp.paths.sampling_trace_plot, dpi=300)

# fig, ax = plot_profiles(result, true_params.to_dict())
# fig.savefig(exp.paths.profile_plot, dpi=300)

# fig, ax = plot_confidence_intervals(result, true_params.to_dict())
# fig.savefig(exp.paths.confidence_intervals_plot, dpi=300)

# plt.close("all")
