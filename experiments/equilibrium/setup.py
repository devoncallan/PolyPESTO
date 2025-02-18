from typing import Dict

import numpy as np
import pandas as pd

from polypesto.core import petab as pet

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
