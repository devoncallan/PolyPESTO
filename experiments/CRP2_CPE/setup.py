from typing import Dict

import numpy as np
import pandas as pd

import polypesto.core.petab as pet
import polypesto.models.cpe as cpe


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
