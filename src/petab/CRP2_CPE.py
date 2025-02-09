from typing import Dict

import numpy as np
import pandas as pd

import src.utils.petab as pet
import src.models.cpe as cpe

##########################
### DEFINED PETAB DATA ###
##########################
### All functions should take no arguments and return a pet.PetabData object


def exp_0() -> pet.PetabData:

    t_eval = list(np.arange(0, 1, 0.1, dtype=float))
    fA0s = np.array([0.25, 0.5, 0.75, 0.1], dtype=float)
    cM0s = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    cond_df = create_conditions_df(fA0s, cM0s)
    obs_df = default_observables()
    param_df = default_parameters()
    empty_meas_df = pet.define_empty_measurements(obs_df, cond_df, t_eval)

    return pet.PetabData(
        obs_df=obs_df, cond_df=cond_df, param_df=param_df, meas_df=empty_meas_df
    )


##############################
### PETAB HELPER FUNCTIONS ###
##############################


def create_ODE_model() -> cpe.CPEModel:
    return cpe.create_model("ODE_CPE", default_observables())


def create_conditions_df(fA0s, cM0s) -> pd.DataFrame:
    return pet.define_conditions(
        {
            "A0": fA0s * cM0s,
            "B0": (1 - fA0s) * cM0s,
        }
    )


############################
### PETAB DEFAULT INPUTS ###
############################


def default_conditions() -> pd.DataFrame:
    return create_conditions_df([1.0], [1.0])


def default_fit_params() -> Dict[str, pet.FitParameter]:

    return {
        "rA": pet.FitParameter(
            id="rA",
            scale=pet.C.LOG10,
            bounds=(1e-2, 1e2),
            nominal_value=1.0,
            estimate=True,
        ),
        "rB": pet.FitParameter(
            id="rB",
            scale=pet.C.LOG10,
            bounds=(1e-2, 1e2),
            nominal_value=1.0,
            estimate=True,
        ),
        "rX": pet.FitParameter(
            id="rX",
            scale=pet.C.LOG10,
            bounds=(1e-3, 1e3),
            nominal_value=1.0,
            estimate=False,
        ),
        "KAA": pet.FitParameter(
            id="KAA",
            scale=pet.C.LIN,
            bounds=(0, 1),
            nominal_value=0.0,
            estimate=False,
        ),
        "KAB": pet.FitParameter(
            id="KAB",
            scale=pet.C.LIN,
            bounds=(0, 1),
            nominal_value=0.0,
            estimate=False,
        ),
        "KBA": pet.FitParameter(
            id="KBA",
            scale=pet.C.LIN,
            bounds=(0, 1),
            nominal_value=0.0,
            estimate=False,
        ),
        "KBB": pet.FitParameter(
            id="KBB",
            scale=pet.C.LIN,
            bounds=(0, 1),
            nominal_value=0.0,
            estimate=False,
        ),
    }


def default_parameters() -> pd.DataFrame:
    return pet.define_parameters(default_fit_params())


def default_observables() -> pd.DataFrame:
    return pet.define_observables({"xA": "xA", "xB": "xB"}, noise_value=0.02)
