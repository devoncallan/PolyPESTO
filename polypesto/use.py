from polypesto.models.sbml.CRP2 import IrreversibleCPE
from polypesto.models.sbml.CRP2 import ReversibleCPE
from polypesto.models.sbml.CRP2 import IrreversibleRxn
from polypesto.models.sbml.CRP2 import ReversibleRxn

FIT_PARAMS_REV_0000 = ReversibleCPE.get_default_fit_params()

FIT_PARAMS_REV_1000 = ReversibleCPE.get_default_fit_params()
FIT_PARAMS_REV_1000["KAA"].estimate = True

FIT_PARAMS_REV_0100 = ReversibleCPE.get_default_fit_params()
FIT_PARAMS_REV_0100["KAB"].estimate = True

FIT_PARAMS_REV_0010 = ReversibleCPE.get_default_fit_params()
FIT_PARAMS_REV_0010["KBA"].estimate = True

FIT_PARAMS_REV_0001 = ReversibleCPE.get_default_fit_params()
FIT_PARAMS_REV_0001["KBB"].estimate = True

FIT_PARAMS_REV_1010 = ReversibleCPE.get_default_fit_params()
FIT_PARAMS_REV_1010["KAA"].estimate = True
FIT_PARAMS_REV_1010["KBA"].estimate = True

FIT_PARAMS_REV_1001 = ReversibleCPE.get_default_fit_params()
FIT_PARAMS_REV_1001["KAA"].estimate = True
FIT_PARAMS_REV_1001["KBB"].estimate = True

FIT_PARAMS_REV_0101 = ReversibleCPE.get_default_fit_params()
FIT_PARAMS_REV_0101["KAB"].estimate = True
FIT_PARAMS_REV_0101["KBB"].estimate = True


"""
Choices:
- Fit Parameters:
    - This defines the model that will be estimated. 
    - Can be different from the true parameters the data was generated from.

- Defining PetabData:
    - Defining the experimental reaction conditions
    - Mostly changing t_eval, fA0s, and cM0s.
    - How do we want to make this easy/simple/organized to define?
    - TODO: What are the most important use cases?

- Selecting True Parameters:
    - Defining how to generate experimental measurement data from the PetabData
    - Maybe sweep 0.1, 0.5, 1.0, 2.0, 10.0 for rA, rB


How do we structure this in a way that is:
- Simple to use
- Easy to understand
- Elegant
- Flexible
- Extensible

Do I need to rework the parameters a bit?
- Rethink true parameter loading and defining.
- Need a workflow to define programatically, not manually.
- Need to decide what parameters to even consider.

Start from the end:
- What do I want to do?
- What do I want the API to look like?
- i.e., how do I want to use this?
- How do I want to access the data?



def standard_library():
    rA = [0.1, 0.5, 1.0, 2.0, 10.0]
    rB = [0.1, 0.5, 1.0, 2.0, 10.0]
    
    for rA in rA:
        for rB in rB:
            pg.add(Params(rA, rB))

    return pg

# Defining and running fitting routine

    fA0s = [[0.1], [0.25], [0.5]]
    cM0s = [[1.0], [1.0], [1.0]]
    t_eval = np.arange(0, 1, 0.1, dtype=float)
    
    for fA0, cM0 in zip(fA0s, cM0s):
        exp = Experiment(standard_library, fA0s=fA0, cM0s=cM0, t_eval=t_eval)
        
        exp.fit(solver_params, )
        
        exp.fit(overwrite=False, save_figs=True) # A lot happens here under the hood.
        
# How do we run this?
# - Dedicated python script?



"""
# Fit Parameters + Defining PetabData + Selecting True Parameters

# true_params = pg.get_parameter_group("")

from dataclasses import dataclass


@dataclass
class CRP2Params:
    rA: float
    rB: float
    rX: float = 1.0
    KAA: float = 0.0
    KAB: float = 0.0
    KBA: float = 0.0
    KBB: float = 0.0
    
class ParamSet:
    
    slow = CRP2Params(rA=10.0, rB=10.0, rX=10.0)
    fast = CRP2Params(rA=100.0, rB=100.0, rX=100.0)


def built_parameter_sets():
    
    # 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
    
    # 2^(n-1) = 2^5 = 32 parameter sets
    
    # 
    # 
    # 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95
    # rA = [0.1, 0.5, 1.0, 2.0]
    # rA_0.10_rB_1.00
    # rA_2.00_rB_10.0
    # rA_0.50_rB_0.50
    # rA_1.00_rB_1.00
    # rA_5.00_rB_20.0
    # rA_10.0_rB_5.00
    # rA_20.0_rB_10.0

    all_sets = {
        "p0": CRP2Params(1.0, 1.0),
        "p1": CRP2Params(2.0, 0.5),
        "p2": CRP2Params(0.5, 2.0),
        "p3": CRP2Params(2.0, 2.0),
        "p4": CRP2Params(0.5, 0.5),
        "p5": CRP2Params(20.0, 10.0),
        "p6": CRP2Params(20.0, 5.0),
        
    }
    
    # Define the parameter sets
    # 
    
    groups = {
        "SLOW": ["p0"],
    }

    pass
