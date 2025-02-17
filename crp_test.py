import numpy as np
from matplotlib import pyplot as plt

# import petab.v1.C as C
# import pypesto
# import pypesto.optimize as optimize
# import pypesto.petab
# import pypesto.sample as sample
# import pypesto.visualize as visualize

from src._petab import CRP2_CPE as crp
from src.utils.pypesto import create_problem_set, load_pypesto_problem
from src.models.sbml import CRP2_CPE
from src.utils.params import ParameterContainer
from src.utils.plot import plot_all_measurements


pc = ParameterContainer.from_json("/PolyPESTO/src/data/parameters/CRP2_CPE.json")
pg = pc.get_parameter_group("IRREVERSIBLE")

paths = create_problem_set(CRP2_CPE, pg, crp.exp_0(), force_compile=False)
