from typing import Dict

import pandas as pd
import petab.v1.C as C

from src.utils import petab as pet
from src.models.sbml import CRP2_CPE
import src.models.amici as am
import src.models.cpe as cpe


def create_amici_model(model_dir: str, **kwargs) -> am.AmiciModel:

    return am.create_model(
        model_def=CRP2_CPE,
        obs_df=default_observables(),
        model_dir=model_dir,
        **kwargs,
    )


def create_ODE_model() -> cpe.CPEModel:
    return cpe.create_model("ODE_CPE", default_observables())


def create_conditions_df(fA0s, cM0s) -> pd.DataFrame:
    return pet.define_conditions(
        {
            "A0": fA0s * cM0s,
            "B0": (1 - fA0s) * cM0s,
        }
    )


def default_params() -> Dict[str, pet.FitParameter]:

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


def default_observables() -> pd.DataFrame:
    return pet.define_observables({"xA": "xA", "xB": "xB"}, noise_value=0.02)


### DEFINE THE MODEL TO FIT ###
# - Define the model (CRP2_CPE)
# - Define the fitting paramters (model) range of values, estimate or not, etc.

### DEFINE THE FITTING ROUTINES ###
# - Define a set of pypesto fitting routines to run

### RUN AND SAVE THE FITTING ROUTINES ###
# - Run pypesto fitting routines
# - Save figures and results


### Tool to compare the results ###
# - Import the results
# - See results from a specific parameter set
# -
