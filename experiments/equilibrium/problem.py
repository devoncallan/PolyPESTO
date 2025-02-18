from typing import Dict, Optional

import numpy as np

import polypesto.core.petab as pet
from .setup import (
    create_conditions_df,
    default_observables,
    default_parameters,
    default_fit_params,
)

##########################
### DEFINED PETAB DATA ###
##########################


def exp_0(params_dict: Optional[Dict[str, pet.FitParameter]] = None) -> pet.PetabData:

    if params_dict is None:
        params_dict = default_fit_params()
    param_df = pet.define_parameters(params_dict)

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


def exp_0(params_dict: Optional[Dict[str, pet.FitParameter]] = None) -> pet.PetabData:

    if params_dict is None:
        params_dict = default_parameters()
    param_df = pet.define_parameters(params_dict)

    pass
