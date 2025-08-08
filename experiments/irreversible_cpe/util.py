from typing import Optional, List
import numpy as np

from polypesto.core.experiment import create_simulation_conditions
from polypesto.core.experiment import Experiment
from polypesto.core.study import Study, create_study
from polypesto.core.params import ParameterGroup

from polypesto.models.CRP2 import IrreversibleCPE


def get_test_study(
    study: Study,
    test_dir: str,
    overwrite: bool = False,
    fA0s: Optional[List[float]] = None,
) -> Study:

    obs_df = IrreversibleCPE.create_observables(
        # observables={"fA": "fA", "fB": "fB", "xA": "xA", "xB": "xB"}, noise_value=0.02
        observables={"xA": "xA", "xB": "xB"},
        noise_value=0.02,
    )

    if fA0s is None:
        fA0s = [
            [0.05],
            [0.1],
            [0.2],
            [0.3],
            [0.4],
            [0.45],
            [0.5],
            [0.55],
            [0.6],
            [0.7],
            [0.8],
            [0.9],
            [0.95],
        ]

    # t_eval = np.arange(0, 1.01, 0.01)/
    # t_eval = np.linspace(0.01, 0.99, 10)
    t_eval = np.linspace(0, 1, 11)
    test_conditions = create_simulation_conditions(
        dict(
            name=["Test_Study"],
            t_eval=[t_eval],
            conditions=dict(
                fA0=[fA0s], cM0=[[1.0] * len(fA0s)]  # only going to work for single exp
            ),
            fit_params=[None],
            noise_level=[0.00],
        )
    )

    test_study = create_study(
        model=IrreversibleCPE,
        simulation_params=study.simulation_params,
        conditions=test_conditions,
        base_dir=test_dir,
        obs_df=obs_df,
        overwrite=overwrite,
    )

    return test_study


def get_standard_simulation_params() -> ParameterGroup:

    return ParameterGroup.from_dict(
        {
            "id": "irreversible kinetics",
            "parameter_sets": {
                "gradient_lg": {
                    "id": "gradient_lg",
                    "parameters": {
                        "rA": {"id": "rA", "value": 10},
                        "rB": {"id": "rB", "value": 0.5},
                    },
                },
                "gradient_sm": {
                    "id": "gradient_sm",
                    "parameters": {
                        "rA": {"id": "rA", "value": 2},
                        "rB": {"id": "rB", "value": 0.5},
                    },
                },
                "blocky_lg": {
                    "id": "blocky_lg",
                    "parameters": {
                        "rA": {"id": "rA", "value": 10},
                        "rB": {"id": "rB", "value": 5},
                    },
                },
                "blocky_sm": {
                    "id": "blocky_sm",
                    "parameters": {
                        "rA": {"id": "rA", "value": 10},
                        "rB": {"id": "rB", "value": 2},
                    },
                },
                "alternating_lg": {
                    "id": "alternating_lg",
                    "parameters": {
                        "rA": {"id": "rA", "value": 0.5},
                        "rB": {"id": "rB", "value": 0.1},
                    },
                },
                "alternating_sm": {
                    "id": "alternating_sm",
                    "parameters": {
                        "rA": {"id": "rA", "value": 0.1},
                        "rB": {"id": "rB", "value": 0.01},
                    },
                },
                "statistical_lg": {
                    "id": "statistical_lg",
                    "parameters": {
                        "rA": {"id": "rA", "value": 2},
                        "rB": {"id": "rB", "value": 1},
                    },
                },
                "statistical_sm": {
                    "id": "statistical_sm",
                    "parameters": {
                        "rA": {"id": "rA", "value": 1},
                        "rB": {"id": "rB", "value": 0.5},
                    },
                },
            },
        }
    )


def get_gradient_lg() -> ParameterGroup:
    return ParameterGroup.from_dict(
        {
            "id": "irreversible kinetics",
            "parameter_sets": {
                "gradient_lg": {
                    "id": "gradient_lg",
                    "parameters": {
                        "rA": {"id": "rA", "value": 10},
                        "rB": {"id": "rB", "value": 0.5},
                    },
                }
            },
        }
    )
