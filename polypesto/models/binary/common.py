from polypesto.models import sbml


def define_irreversible_k(model: sbml.Model, kpAA_constant=False):

    sbml.create_parameter(model, "kpAA", value=1, constant=kpAA_constant)
    sbml.create_parameter(model, "kpAB", value=1)
    sbml.create_parameter(model, "kpBA", value=1)
    sbml.create_parameter(model, "kpBB", value=1)

    sbml.create_parameter(model, "rA", value=1)
    sbml.create_parameter(model, "rB", value=1)
    sbml.create_parameter(model, "rX", value=1)

    sbml.create_rule(model, "kpAB", formula="kpAA / rA")
    sbml.create_rule(model, "kpBB", formula="kpAA / rX")
    sbml.create_rule(model, "kpBA", formula="kpBB / rB")


def define_reversible_k(model: sbml.Model, **kwargs):

    define_irreversible_k(model, **kwargs)

    sbml.create_parameter(model, "kdAA", value=0)
    sbml.create_parameter(model, "kdAB", value=0)
    sbml.create_parameter(model, "kdBA", value=0)
    sbml.create_parameter(model, "kdBB", value=0)

    sbml.create_parameter(model, "KAA", value=0)
    sbml.create_parameter(model, "KAB", value=0)
    sbml.create_parameter(model, "KBA", value=0)
    sbml.create_parameter(model, "KBB", value=0)

    sbml.create_rule(model, "kdAA", formula=f"kpAA*KAA")
    sbml.create_rule(model, "kdAB", formula=f"kpAB*KAB")
    sbml.create_rule(model, "kdBA", formula=f"kpBA*KBA")
    sbml.create_rule(model, "kdBB", formula=f"kpBB*KBB")
