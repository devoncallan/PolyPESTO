import numpy as np
import pytest


@pytest.fixture(
    params=[
        "BinaryIrreversible",
        "LotkaVolterra",
    ]
)
def model_problem(request: pytest.FixtureRequest, tmp_path: pytest.TempPathFactory):

    model_name = request.param

    if model_name == "BinaryIrreversible":
        from polypesto.models.binary import BinaryIrreversible

        model = BinaryIrreversible(
            observables=["xA", "fA", "FA"],
        )

        true_params = {"rA": 2.0, "rB": 1.0}
        sim_conds = dict(
            A0=[0.70, 0.50],
            B0=[0.30, 0.50],
        )
        t_evals = np.arange(0.05, 0.61, 0.05)
        noise_levels = 0.05

    elif model_name == "LotkaVolterra":
        from polypesto.models.example import LotkaVolterra

        model = LotkaVolterra(
            observables=["x", "y"],
        )

        true_params = {"alpha": 0.5, "beta": 0.025, "delta": 0.01, "gamma": 0.5}
        sim_conds = dict(
            x=[40, 50],
            y=[9, 10],
        )
        t_evals = np.arange(0, 15.1, 0.5)
        noise_levels = 2.0

    return {
        "model": model,
        "true_params": true_params,
        "sim_conds": sim_conds,
        "t_evals": t_evals,
        "noise_levels": noise_levels,
        "data_dir": tmp_path / f"test_{model_name}",
    }
