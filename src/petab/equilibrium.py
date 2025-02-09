from typing import Dict

import numpy as np
import pandas as pd
import petab.v1.C as C

from src.utils import petab as pet

##########################
### DEFINED PETAB DATA ###
##########################
### All functions should take no arguments and return a pet.PetabData object


def exp_0() -> pet.PetabData:

    t_eval = list(np.arange(0, 5, 0.5, dtype=float))
    A0s = np.array([1.0, 0.2, 0.5, 0.8], dtype=float)
    B0s = np.array([0.0, 0.8, 0.5, 0.2], dtype=float)

    cond_df = create_conditions_df(A0s, B0s)
    obs_df = default_observables()
    param_df = default_parameters()
    empty_meas_df = pet.define_empty_measurements(obs_df, cond_df, t_eval)

    return pet.PetabData(
        obs_df=obs_df, cond_df=cond_df, param_df=param_df, meas_df=empty_meas_df
    )


def exp_1() -> pet.PetabData:
    pass


##############################
### PETAB HELPER FUNCTIONS ###
##############################


def create_conditions_df(A, B) -> pd.DataFrame:
    return pet.define_conditions(
        {
            "A": A,
            "B": B,
        }
    )


############################
### PETAB DEFAULT INPUTS ###
############################


def default_conditions() -> pd.DataFrame:
    return create_conditions_df([1], [1])


def default_fit_params() -> Dict[str, pet.FitParameter]:
    bounds = (1e-3, 1e3)
    return {
        "k1": pet.FitParameter(
            id="k1",
            scale=pet.C.LOG10,
            bounds=bounds,
            nominal_value=0.5,
            estimate=True,
        ),
        "k2": pet.FitParameter(
            id="k2",
            scale=pet.C.LOG10,
            bounds=bounds,
            nominal_value=0.5,
            estimate=True,
        ),
    }


def default_parameters() -> pd.DataFrame:
    return pet.define_parameters(default_fit_params())


def default_observables() -> pd.DataFrame:
    return pet.define_observables({"A": "A", "B": "B"}, noise_value=0.02)
