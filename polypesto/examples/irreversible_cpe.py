def main():
    from polypesto.utils._patches import apply as _apply_patches

    _apply_patches()

    import numpy as np

    from polypesto.core.params import ParameterGroup
    from polypesto.core.study import Study, create_study
    from polypesto.core.problem import create_simulation_conditions
    from polypesto.models.CRP2 import IrreversibleCPE
    from polypesto.utils.paths import setup_data_dirs
    from polypesto.visualization import (
        plot_all_results,
        plot_all_ensemble_predictions,
        plot_all_comparisons_1D,
    )

    DATA_DIR, TEST_DIR = setup_data_dirs(__file__)

    simulation_params = ParameterGroup.create_parameter_grid(
        {
            "rA": [0.1],
            "rB": [0.1],
        },
        filter_fn=lambda p: p["rA"] >= p["rB"],
    )

    # Define fitting parameters
    fit_params = IrreversibleCPE.get_default_parameters()
    obs_df = IrreversibleCPE.create_observables(
        observables={"fA": "fA", "fB": "fB", "xA": "xA", "xB": "xB"}, noise_value=0.02
    )
    
    

    # Define experimental configurations
    t_eval = np.arange(0, 1, 0.05)
    fA0s = [[0.1]]
    cM0s = [[1.0]]
    names = [f"fA0_{fA0}_cM0_{cM0}" for fA0, cM0 in zip(fA0s, cM0s)]
    ntrials = len(fA0s)
    assert len(fA0s) == len(cM0s), "fA0s and cM0s must have the same length"

    conditions = create_simulation_conditions(
        dict(
            name=names,
            t_eval=[t_eval] * ntrials,
            conditions=dict(fA0=fA0s, cM0=cM0s),
            fit_params=[fit_params] * ntrials,
            noise_level=[0.02] * ntrials,
        )
    )
    
    

    # Create the study - this will simulate all experiments
    study = create_study(
        model=IrreversibleCPE,
        simulation_params=simulation_params,
        conditions=conditions,
        obs_df=obs_df,
        base_dir=DATA_DIR,
        overwrite=True,
    )

    # Run parameter estimation
    study.run_parameter_estimation(
        config=dict(
            optimize=dict(n_starts=50, method="Nelder-Mead"),
            profile=dict(method="Nelder-Mead"),
            sample=dict(n_samples=10000, n_chains=3),
        ),
        overwrite=False,
    )

    plot_all_comparisons_1D(study)
    plot_all_results(study)


if __name__ == "__main__":
    main()
