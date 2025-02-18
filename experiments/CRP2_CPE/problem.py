from typing import Dict, Optional

import numpy as np
import pandas as pd

import polypesto.core.petab as pet
import polypesto.models.cpe as cpe
from .setup import (
    create_conditions_df,
    default_observables,
    default_parameters,
    default_fit_params,
)

############################
### DEFINE PETAB PROBLEM ###
############################
"""

"""


def ds_0(params_dict: Optional[Dict[str, pet.FitParameter]] = None) -> pet.PetabData:

    if params_dict is None:
        params_dict = default_fit_params()
    param_df = pet.define_parameters(params_dict)

    t_eval = list(np.arange(0, 1, 0.1, dtype=float))
    fA0s = np.array([0.25, 0.5, 0.75, 0.1], dtype=float)
    cM0s = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    cond_df = create_conditions_df(fA0s, cM0s)
    obs_df = default_observables()
    empty_meas_df = pet.define_empty_measurements(obs_df, cond_df, t_eval)

    return pet.PetabData(
        obs_df=obs_df, cond_df=cond_df, param_df=param_df, meas_df=empty_meas_df
    )


def ds_1(params_dict: Optional[Dict[str, pet.FitParameter]] = None) -> pet.PetabData:

    t_eval = list(np.arange(0, 1, 0.1, dtype=float))
    fA0s = np.array([0.25, 0.5, 0.75, 0.1], dtype=float)
    cM0s = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    cond_df = create_conditions_df(fA0s, cM0s)
    obs_df = default_observables()
    fit_params = default_fit_params()
    fit_params["KAA"].estimate = True
    fit_params["KBA"].estimate = True
    param_df = pet.define_parameters(fit_params)
    empty_meas_df = pet.define_empty_measurements(obs_df, cond_df, t_eval)

    return pet.PetabData(
        obs_df=obs_df, cond_df=cond_df, param_df=param_df, meas_df=empty_meas_df
    )
