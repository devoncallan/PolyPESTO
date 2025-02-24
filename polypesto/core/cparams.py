REQUIRED_PARAMS = ["rA", "rB", "rX"]
OBSERVABLES = ["A", "B"]


from .params import Parameter, ParameterSet

# ParameterSet("Param", )
ParameterSet("Fast Kinetics", [1.0, 1.0, 1.0])

from dataclasses import dataclass

@dataclass
class IrrevParams:
    rA: float
    rB: float
    rX: float

param_sets_dict = {}
param_sets_dict["p1"] = IrrevParams(1.0, 1.0, 1.0)
param_sets_dict["p2"] = [2.0, 2.0, 2.0]
param_sets_dict["p3"] = [3.0, 3.0, 3.0]

param_groups_dict = {}
param_groups_dict["SLOW"] = ["p1", "p2"]
param_groups_dict["FAST"] = ["p3"]


class Conditions:
    
    @staticmethod
    def range_of_conditions():
        pass
    
# class DefineParameters:
    