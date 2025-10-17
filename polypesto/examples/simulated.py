#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polypesto.core import create_sim_conditions, simulate_problem
from polypesto.core.pypesto import calculate_cis, create_ensemble, predict_with_ensemble
from polypesto.examples.base import output_dirs

# Model specific imports
from polypesto.models.binary import BinaryIrreversible
from polypesto.models.binary.utils import create_ensemble_pred_problem
from polypesto.vis import plot_ensemble_predictions

OUTPUT_DIR = output_dirs(Path(__file__).stem)


def main():

    # Initialize model with observables
    model = BinaryIrreversible(
        observables=["xA", "FA"], obs_noise={"xA": 0.01, "FA": 0.01}
    )

    # Define true parameters and simulation conditions
    true_params = {"rA": 2.0, "rB": 1.0}
    sim_conds = create_sim_conditions(
        true_params=true_params,
        conds=dict(
            A0=[0.70],
            B0=[0.30],
        ),
        t_evals=np.arange(0.05, 0.61, 0.05),
        meas_noise=0.01,
    )

    # Simulate problem and create parameter estimation problem
    problem = simulate_problem(
        prob_dir=OUTPUT_DIR,
        model=model,
        conds=sim_conds,
        overwrite=True,
    )

    # Run parameter estimation (optimization + sampling)
    result = problem.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=True,
    )
    if result is None:
        return

    # Predict using parameter ensemble from sampling
    ens_prob = create_ensemble_pred_problem(problem.paths.ensemble_dir, model)
    ens, ens_pred = problem.ensemble_prediction(ens_prob)
    
    ens_pred.prediction_results[0].write_to_csv(str(OUTPUT_DIR / "words.csv"))

    if ens is None or ens_pred is None:
        return

    print(f"\n==== Ensemble prediction results ====")
    print("Check identifiability:")
    id_df = ens.check_identifiability()
    id_df.to_csv(OUTPUT_DIR / "identifiability.csv")
    print(id_df)
    print("Compute ensemble summary:")
    print(ens.compute_summary(percentiles_list=(5, 25, 50, 75, 95)))
    

    from typing import Dict
    from pypesto.result import PredictionResult
    from pypesto.visualize import projection_scatter_umap, projection_scatter_pca, sampling_prediction_trajectories

    from pypesto.ensemble import get_umap_representation_parameters, get_umap_representation_predictions
    from pypesto.ensemble import get_pca_representation_parameters, get_pca_representation_predictions
    from pypesto.ensemble import get_covariance_matrix_parameters, get_covariance_matrix_predictions
    from pypesto.ensemble import get_spectral_decomposition_parameters, get_spectral_decomposition_predictions

    pca_params = get_pca_representation_parameters(ens)
    print("PCA parameter representation:", pca_params)
    pca_preds = get_pca_representation_predictions(ens_pred)
    print("PCA prediction representation:", pca_preds)

    cov_matrix_params = get_covariance_matrix_parameters(ens)
    print("Covariance matrix parameter representation:", cov_matrix_params)
    cov_matrix_preds = get_covariance_matrix_predictions(ens_pred)
    print("Covariance matrix prediction representation:", cov_matrix_preds)

    # umap_params = get_umap_representation_parameters(ens)
    # print("UMAP parameter representation:", umap_params)
    # # print(umap_params)

    # umap_preds = get_umap_representation_predictions(ens)
    # print("UMAP prediction representation:", umap_preds)
    # print(umap_preds)

    return

    pca_params = get_pca_representation_parameters(ens)
    pca_preds = get_pca_representation_predictions(ens_pred)

    cov_matrix_params = get_covariance_matrix_parameters(ens)
    cov_matrix_preds = get_covariance_matrix_predictions(ens_pred)

    print("Compute ensemble prediction summary:")
    pred_results: Dict[str, PredictionResult] = ens_pred.compute_summary(percentiles_list=(5, 25, 50, 75, 95))
    pred_results["mean"].write_to_csv(str(OUTPUT_DIR / "ensemble_prediction_summary.csv"))
    # print(ens_pred.compute_summary(percentiles_list=(5, 25, 50, 75, 95)))

    return
    print("Compute chi2 values:")
    print(ens_pred.compute_chi2(ens_prob.pypesto_problem.objective))

    calculate_cis(result, ci_level=0.95)


if __name__ == "__main__":
    main()
