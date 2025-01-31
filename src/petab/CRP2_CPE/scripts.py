from src.utils import petab as pet
from src.models.sbml import CRP2_CPE
import src.models.amici as am
import src.models.cpe as cpe
import pandas as pd

from src.utils.params import ParameterContainer
from src.petab.dataset import PetabDataset
from src.utils.plot import plot_all_measurements

import petab.v1.C as C

import numpy as np
from typing import Dict, Any, List
import os

#####################################
### DEFINING THE FITTING DATASETS ###
#####################################


def create_CRP2_CPE_Model(model_dir: str, force_compile=False) -> am.AmiciModel:

    model = am.create_model(
        sbml_model_func=CRP2_CPE,
        obs_df=pet.define_observables({"xA": "xA", "xB": "xB"}, noise_value=0.02),
        model_dir=model_dir,
        force_compile=force_compile,
    )
    return model


def create_CRP2_CPE_conditions(fA0s, cM0s) -> pd.DataFrame:
    return pet.define_conditions(
        {
            "A0": fA0s * cM0s,
            "B0": (1 - fA0s) * cM0s,
        }
    )


def create_CPE_Model() -> cpe.CPEModel:
    # Call cpe.create_model() with the appropriate arguments
    name = "ODE_CPE"
    obs_df=pet.define_observables({"xA": "xA", "xB": "xB"}, noise_value=0.02)
    model = cpe.create_model(name, obs_df)

    return model


def gen_dataset() -> PetabDataset:

    # Create the model
    model_dir = "/PolyPESTO/src/petab/CRP2_CPE/"
    model = create_CRP2_CPE_Model(model_dir=model_dir, force_compile=False)

    # Define a set of parameters to sweep (e.g. irreversible params, all params, base set of params, extended set, etc.)
    pc = ParameterContainer.from_json("/PolyPESTO/src/data/parameters/CRP2_CPE.json")
    pg = pc.get_parameter_group("IRREVERSIBLE")

    # Define a set of conditions to generate synthetic data
    t_eval = list(np.arange(0, 1, 0.1, dtype=float))
    fA0s = np.array([0.25, 0.5, 0.75], dtype=float)
    cM0s = np.array([1.0, 1.0, 1.0], dtype=float)
    cond_df = create_CRP2_CPE_conditions(fA0s, cM0s)

    # Generate and save the dataset
    ds_name = "ds_0"
    # ds_dir = f"/PolyPESTO/src/data_sim/CRP2_CPE/{ds_name}"
    ds_dir = f"/PolyPESTO/src/data/datasets/CRP2_CPE/{ds_name}"

    ds = model.generate_dataset(
        param_group=pg, t_eval=t_eval, cond_df=cond_df, name=ds_name
    )  # .write(ds_dir)

    meas_dfs = ds.get_meas_dfs()
    for id, df in meas_dfs.items():
        print(f"Parameter set {id}")
        # print(df)
        plot_all_measurements(df)
    # Load the dataset
    # ds1 = PetabDataset.load(ds_dir)

    return ds


# def exp_0() -> None:

#     # ds = generate_dataset()
#     ds = load_dataset("NAME")

#     fit_params = default_fit_params()
#     fit_params["KAA"].estimate = True

#     write_yaml_file(
#         ds,
#         fit_params,
#         "/PolyPESTO/src/data_sim/CRP2_CPE/",
#     )

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


# def default_fit_params():
#     return {
#         "rA": FitParameter(
#             id="rA",
#             scale=pet.C.LOG10,
#             bounds=(1e-2, 1e2),
#             nominal_value=1.0,
#             estimate=True,
#         ),
#         "rB": FitParameter(
#             id="rB",
#             scale=pet.C.LOG10,
#             bounds=(1e-2, 1e2),
#             nominal_value=1.0,
#             estimate=True,
#         ),
#         "rX": FitParameter(
#             id="rX",
#             scale=pet.C.LOG10,
#             bounds=(1e-3, 1e3),
#             nominal_value=1.0,
#             estimate=False,
#         ),
#         "KAA": FitParameter(
#             id="KAA",
#             scale=pet.C.LIN,
#             bounds=(0, 1),
#             nominal_value=0.0,
#             estimate=False,
#         ),
#         "KAB": FitParameter(
#             id="KAB",
#             scale=pet.C.LIN,
#             bounds=(0, 1),
#             nominal_value=0.0,
#             estimate=False,
#         ),
#         "KBA": FitParameter(
#             id="KBA",
#             scale=pet.C.LIN,
#             bounds=(0, 1),
#             nominal_value=0.0,
#             estimate=False,
#         ),
#         "KBB": FitParameter(
#             id="KBB",
#             scale=pet.C.LIN,
#             bounds=(0, 1),
#             nominal_value=0.0,
#             estimate=False,
#         ),
#     }
