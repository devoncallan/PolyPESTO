from typing import List

import libsbml

from polypesto.models import sbml


######################################################
### Controlled Radical Polymerization (CRP) models ###
######################################################


def define_irreversible_k(
    model: sbml.Model, kpAA_constant=False
) -> List[libsbml.Parameter]:

    print("Creating irreversible CRP parameters.")
    kpAA = sbml.create_parameter(model, "kpAA", value=1, constant=kpAA_constant)
    kpAB = sbml.create_parameter(model, "kpAB", value=1)
    kpBA = sbml.create_parameter(model, "kpBA", value=1)
    kpBB = sbml.create_parameter(model, "kpBB", value=1)

    rA = sbml.create_parameter(model, "rA", value=1)
    rB = sbml.create_parameter(model, "rB", value=1)
    rX = sbml.create_parameter(model, "rX", value=1)

    sbml.create_rule(model, kpAB, formula=f"{kpAA.getId()} / {rA.getId()}")
    sbml.create_rule(model, kpBB, formula=f"{kpAA.getId()} / {rX.getId()}")
    sbml.create_rule(model, kpBA, formula=f"{kpBB.getId()} / {rB.getId()}")

    return [kpAA, kpAB, kpBA, kpBB]


def define_reversible_k(model: sbml.Model, **kwargs) -> List[libsbml.Parameter]:

    [kpAA, kpAB, kpBA, kpBB] = define_irreversible_k(model, **kwargs)

    print("Creating reversile CRP parameters.")
    kdAA = sbml.create_parameter(model, "kdAA", value=1)
    kdAB = sbml.create_parameter(model, "kdAB", value=1)
    kdBA = sbml.create_parameter(model, "kdBA", value=1)
    kdBB = sbml.create_parameter(model, "kdBB", value=1)

    KAA = sbml.create_parameter(model, "KAA", value=0)
    KAB = sbml.create_parameter(model, "KAB", value=0)
    KBA = sbml.create_parameter(model, "KBA", value=0)
    KBB = sbml.create_parameter(model, "KBB", value=0)

    sbml.create_rule(model, kdAA, formula=f"{kpAA.getId()} * {KAA.getId()}")
    sbml.create_rule(model, kdAB, formula=f"{kpAB.getId()} * {KAB.getId()}")
    sbml.create_rule(model, kdBA, formula=f"{kpBA.getId()} * {KBA.getId()}")
    sbml.create_rule(model, kdBB, formula=f"{kpBB.getId()} * {KBB.getId()}")

    return [kpAA, kpAB, kpBA, kpBB, kdAA, kdAB, kdBA, kdBB]
