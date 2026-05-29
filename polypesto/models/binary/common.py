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


def define_Lowry_I(model: sbml.Model, **kwargs):

    define_irreversible_k(model, **kwargs)

    sbml.create_parameter(model, "kdAA", value=0)
    sbml.create_parameter(model, "KAA", value=0)

    sbml.create_rule(model, "kdAA", formula=f"kpAA*KAA")

def define_Lowry_I_Temp(model: sbml.Model, **kwargs):

    define_irreversible_k(model, **kwargs)

    sbml.create_parameter(model, "kdAA", value=0)
    sbml.create_parameter(model, "KAA", value=0)

    sbml.create_rule(model, "KAA", formula=f"exp(6.21 - 1898.0/T_K)")
    sbml.create_rule(model, "kdAA", formula=f"kpAA*KAA")


def define_Lowry_I_Temp_Fit(model: sbml.Model, Tref: float = 350.0, **kwargs):
    """
    Same as define_Lowry_I_Temp but with KAA(T) parameterised in terms of two
    fittable parameters (KAA_Tref, dH_R) instead of the hard-coded Tsarevsky
    constants. Reparameterised at a reference temperature Tref so the joint
    prior on (KAA_Tref, dH_R) can be approximated by independent normals.

        KAA(T) = KAA_Tref * exp( -(dH_R) * (1/T - 1/Tref) )

    Equivalent to KAA = exp(dS_R - dH_R/T) with dS_R = ln(KAA_Tref) + dH_R/Tref.
    Tref is chosen near the median of the calibration data so KAA_Tref is
    well-determined (residual ~0.04 mol/L SD at Tref=350 K from 9 Tsarevsky
    points).
    """
    define_irreversible_k(model, **kwargs)

    sbml.create_parameter(model, "kdAA", value=0)
    sbml.create_parameter(model, "KAA", value=0)
    sbml.create_parameter(model, "KAA_Tref", value=2.27)
    sbml.create_parameter(model, "dH_R", value=1893.0)
    sbml.create_parameter(model, "Tref", value=float(Tref), constant=True)

    sbml.create_rule(
        model,
        "KAA",
        formula="KAA_Tref * exp(-dH_R * (1/T_K - 1/Tref))",
    )
    sbml.create_rule(model, "kdAA", formula="kpAA*KAA")


def define_Lowry_II_Temp_Fit(model: sbml.Model, Tref: float = 350.0, **kwargs):
    """
    Lowry Case II: both AA and BA dyads depropagate. Van't Hoff K_AA(T) with
    fittable (KAA_Tref, dH_R), plus a fittable ratio f_BA = KBA/KAA. With
    f_BA = 0 this reduces exactly to define_Lowry_I_Temp_Fit.

        KAA(T) = KAA_Tref * exp(-dH_R * (1/T - 1/Tref))
        KBA    = f_BA * KAA
        kdAA   = kpAA * KAA
        kdBA   = kpBA * KBA
    """
    define_irreversible_k(model, **kwargs)

    sbml.create_parameter(model, "kdAA", value=0)
    sbml.create_parameter(model, "kdBA", value=0)
    sbml.create_parameter(model, "KAA", value=0)
    sbml.create_parameter(model, "KBA", value=0)
    sbml.create_parameter(model, "KAA_Tref", value=2.27)
    sbml.create_parameter(model, "dH_R", value=1893.0)
    sbml.create_parameter(model, "f_BA", value=0.0)
    sbml.create_parameter(model, "Tref", value=float(Tref), constant=True)

    sbml.create_rule(
        model,
        "KAA",
        formula="KAA_Tref * exp(-dH_R * (1/T_K - 1/Tref))",
    )
    sbml.create_rule(model, "KBA", formula="f_BA * KAA")
    sbml.create_rule(model, "kdAA", formula="kpAA*KAA")
    sbml.create_rule(model, "kdBA", formula="kpBA*KBA")
