from typing import Dict

import numpy as np
import pandas as pd

import polypesto.core.petab as pet

##############################
### PETAB HELPER FUNCTIONS ###
##############################


def create_conditions_df(x0, y0) -> pd.DataFrame:
    return pet.define_conditions({"x": x0, "y": y0})


############################
### PETAB DEFAULT INPUTS ###
############################


def default_conditions() -> pd.DataFrame:
    return create_conditions_df([5, 10, 20], [10, 20, 30])


def default_fit_params() -> Dict[str, pet.FitParameter]:
    return {
        "alpha": pet.FitParameter(
            id="alpha",
            scale=pet.C.LOG10,
            bounds=(1e-1, 1e1),
            nominal_value=2.0,
            estimate=True,
        ),
        "beta": pet.FitParameter(
            id="beta",
            scale=pet.C.LOG10,
            bounds=(1e-1, 1e1),
            nominal_value=0.4,
            estimate=True,
        ),
        "delta": pet.FitParameter(
            id="delta",
            scale=pet.C.LOG10,
            bounds=(1e-1, 1e1),
            nominal_value=0.4,
            estimate=True,
        ),
        "gamma": pet.FitParameter(
            id="gamma",
            scale=pet.C.LOG10,
            bounds=(1e-1, 1e1),
            nominal_value=0.3,
            estimate=True,
        ),
    }


def default_parameters() -> pd.DataFrame:
    return pet.define_parameters(default_fit_params())


def default_observables() -> pd.DataFrame:
    return pet.define_observables({"x": "x", "y": "y"}, noise_value=0.02)
