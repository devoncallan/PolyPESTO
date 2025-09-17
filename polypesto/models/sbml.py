from pathlib import Path
from typing import Dict, Tuple, TypeAlias, Optional
import os
import time

import libsbml
from petab.v1.models.sbml_model import SbmlModel

ModelDefinition: TypeAlias = SbmlModel
Document: TypeAlias = libsbml.SBMLDocument
Model: TypeAlias = libsbml.Model

SBML_LEVEL = 3
SBML_VERSION = 2


def write_model(model_def: ModelDefinition, model_filepath: str) -> None:
    """Writes an SBML model from the given file path."""

    print(f"Writing SBML model ({model_def.model_id}) to {model_filepath}")
    model_path = Path(model_filepath)
    os.makedirs(model_path.parent, exist_ok=True)
    model_def.to_file(model_path)

    validateSBML().validate(model_path)


#####################################
### SBML model creation functions ###
#####################################


def _check(value, message: str):
    """If 'value' is None, prints an error message constructed using
    'message' and then exits with status code 1.  If 'value' is an integer,
    it assumes it is a libSBML return status code.  If the code value is
    LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
    prints an error message constructed using 'message' along with text from
    libSBML explaining the meaning of the code, and exits with status code 1.
    """
    if value == None:
        raise SystemExit("LibSBML returned a null value trying to " + message + ".")
    elif type(value) is int:
        if value == libsbml.LIBSBML_OPERATION_SUCCESS:
            return
        else:
            err_msg = (
                "Error encountered trying to "
                + message
                + "."
                + "LibSBML returned error code "
                + str(value)
                + ': "'
                + libsbml.OperationReturnValue_toString(value).strip()
                + '"'
            )
            raise SystemExit(err_msg)
    else:
        return


def init_model(name: str) -> Tuple[Document, Model]:

    try:
        document = Document(SBML_LEVEL, SBML_VERSION)
    except ValueError:
        raise SystemExit("Could not create SBMLDocumention object")

    model: Model = document.createModel()
    _check(model, "create model")
    _check(model.setName(name), "set model name")
    _check(model.setTimeUnits("second"), "set model-wide time units")
    _check(model.setExtentUnits("mole"), "set model units of extent")
    _check(model.setSubstanceUnits("mole"), "set model substance units")

    per_second: libsbml.UnitDefinition = model.createUnitDefinition()
    _check(per_second, "create unit definition")
    _check(per_second.setId("per_second"), "set unit definition id")

    unit: libsbml.Unit = per_second.createUnit()
    _check(unit, "create unit on per_second")
    _check(unit.setId("per_second"), "set unit id")
    _check(unit.setKind(libsbml.UNIT_KIND_SECOND), "set unit kind")
    _check(unit.setExponent(-1), "set unit exponent")
    _check(unit.setScale(0), "set unit scale")
    _check(unit.setMultiplier(1), "set unit multiplier")

    return document, model


def create_model(model: Model, document: Document) -> ModelDefinition:

    return ModelDefinition(
        sbml_model=model, sbml_document=document, model_id=model.getId()
    )


def create_compartment(
    model: Model,
    id: str,
    size: float = 1.0,
    spatialDimensions: int = 3,
    units: str = "litre",
    isConstant=True,
) -> libsbml.Compartment:

    c: libsbml.Compartment = model.createCompartment()
    _check(c, "create compartment")
    _check(c.setId(id), "set compartment id")
    _check(c.setConstant(isConstant), 'set compartment "constant"')
    _check(c.setSize(size), 'set compartment "size"')
    _check(c.setSpatialDimensions(spatialDimensions), "set compartment dimensions")
    _check(c.setUnits(units), "set compartment size units")

    return c


def create_species(
    model: Model,
    id: str,
    initialAmount: Optional[float] = 0.0,
    constant=False,
    units: str = "dimensionless",
    boundaryCondition: bool = False,
) -> libsbml.Species:

    s1: libsbml.Species = model.createSpecies()
    c: libsbml.Compartment = model.getCompartment(0)

    _check(s1, "create species s1")
    _check(s1.setId(id), "set species s1 id")
    _check(s1.setCompartment(c.getId()), "set species s1 compartment")
    _check(s1.setConstant(constant), 'set "constant" attribute on s1')
    if initialAmount is not None:
        _check(s1.setInitialAmount(initialAmount), "set initial amount for s1")
    _check(s1.setSubstanceUnits(units), "set substance units for s1")
    _check(s1.setBoundaryCondition(boundaryCondition), 'set "boundaryCondition" on s1')
    _check(s1.setHasOnlySubstanceUnits(False), 'set "hasOnlySubstanceUnits" on s1')

    return s1


def create_parameter(
    model: Model,
    id: str,
    value: float = 0.0,
    constant: bool = False,
    units: str = "dimensionless",
) -> libsbml.Parameter:

    k: libsbml.Parameter = model.createParameter()

    _check(k, "create parameter k")
    _check(k.setId(id), "set parameter k id")
    _check(k.setConstant(constant), 'set parameter k "constant"')
    _check(k.setValue(value), "set parameter k value")
    _check(k.setUnits(units), "set parameter k units")

    return k


def create_initial_assignment(
    model: Model, variable_id: str, formula: str
) -> libsbml.InitialAssignment:
    """
    Create an initial assignment in the SBML model.

    Parameters:
    - model: The SBML model to which the initial assignment will be added.
    - variable_id: The ID of the parameter or species for which the initial assignment is defined.
    - formula: The formula representing the initial assignment.

    Returns:
    - InitialAssignment: The created initial assignment object.
    """

    initial_assignment: libsbml.InitialAssignment = model.createInitialAssignment()

    _check(initial_assignment, "create initial assignment")
    _check(initial_assignment.setSymbol(variable_id), "set initial assignment variable")
    math_ast: libsbml.ASTNode = libsbml.parseL3Formula(formula)
    _check(math_ast, "create AST for initial assignment formula")
    _check(initial_assignment.setMath(math_ast), "set math on initial assignment")

    return initial_assignment


def create_reaction(
    model: Model,
    id: str,
    reactantsDict: Dict[str, int],
    productsDict: Dict[str, int],
    kineticLaw: str,
) -> libsbml.Reaction:

    r: libsbml.Reaction = model.createReaction()
    _check(r, "create reaction")
    _check(r.setId(id), "set reaction id")
    _check(r.setReversible(False), "set reaction reversibility flag")

    for reactant, stoich in reactantsDict.items():
        species_ref1: libsbml.SpeciesReference = r.createReactant()
        _check(species_ref1, "create reactant")
        _check(species_ref1.setSpecies(reactant), "assign reactant species")
        _check(species_ref1.setConstant(False), 'set "constant" on species ref 1')
        _check(
            species_ref1.setStoichiometry(stoich), "set stoichiometry on species ref 1"
        )

    for product, stoich in productsDict.items():
        species_ref2: libsbml.SpeciesReference = r.createProduct()
        _check(species_ref2, "create product")
        _check(species_ref2.setSpecies(product), "assign product species")
        _check(species_ref2.setConstant(False), 'set "constant" on species ref 2')
        _check(
            species_ref2.setStoichiometry(stoich), "set stoichiometry on species ref 2"
        )

    # Create kinetic law
    c1: libsbml.Compartment = model.getCompartment(0)

    kineticLaw = f"{kineticLaw} * {c1.getId()}"
    kin_law: libsbml.KineticLaw = r.createKineticLaw()
    math_ast: libsbml.ASTNode = libsbml.parseL3Formula(kineticLaw)

    _check(math_ast, "create AST for rate expression")
    _check(kin_law, "create kinetic law")
    _check(kin_law.setMath(math_ast), "set math on kinetic law")

    return r


def create_rule(model: Model, var_id: str, formula: str = "") -> libsbml.AssignmentRule:

    rule: libsbml.AssignmentRule = model.createAssignmentRule()
    _check(rule.setVariable(var_id), "set variable")

    # math_ast: ASTNode = parseL3Formula(f'{species1.getId()} + {species2.getId()} + 1e-10')
    math_ast: libsbml.ASTNode = libsbml.parseL3Formula(formula)
    _check(math_ast, "create AST for rate expression")
    _check(rule.setMath(math_ast), "set math on kinetic law")

    return rule


def create_rate_rule(model: Model, var_id: str, formula: str = "") -> libsbml.RateRule:

    rate_rule: libsbml.RateRule = model.createRateRule()
    _check(rate_rule.setVariable(var_id), "set variable")

    # math_ast: ASTNode = parseL3Formula(f'{species1.getId()} + {species2.getId()} + 1e-10')
    math_ast: libsbml.ASTNode = libsbml.parseL3Formula(formula)
    _check(math_ast, "create AST for rate expression")
    _check(rate_rule.setMath(math_ast), "set math on kinetic law")

    return rate_rule


def create_algebraic_rule(model: Model, formula: str = "") -> libsbml.AlgebraicRule:

    rule: libsbml.AlgebraicRule = model.createAlgebraicRule()
    math_ast: libsbml.ASTNode = libsbml.parseL3Formula(formula)
    _check(math_ast, "create AST for rate expression")
    _check(rule.setMath(math_ast), "set math on kinetic law")

    return rule


def add_termination_event(model: Model, formula: str = "") -> libsbml.Event:
    """
    Adds an event to terminate the simulation when dA or dB is negative.

    Parameters:
    - model: The SBML model to which the event will be added.
    - dA_id: The ID of the parameter representing dA.
    - dB_id: The ID of the parameter representing dB.
    """
    # Create the event
    termination_event: libsbml.Event = model.createEvent()
    _check(termination_event, "create termination event")
    # termination_event.

    # Define the trigger for the event based on the formula
    trigger: libsbml.Trigger = termination_event.createTrigger()
    # trigger.
    _check(trigger, "create event trigger")
    _check(trigger.setMath(libsbml.parseL3Formula(formula)), "set trigger condition")
    _check(trigger.setPersistent(True), "set trigger persistent")
    _check(trigger.setInitialValue(True), "set trigger initial value")

    # Define the action for the event: stop the simulation
    # Stop immediately when the condition is met
    termination_event.setUseValuesFromTriggerTime(True)

    return termination_event


def create_event_assignment(
    model: Model, event_id: str, variable_id: str, formula: str
) -> libsbml.EventAssignment:
    """
    Create an event assignment in the SBML model.

    Parameters:
    - model: The SBML model to which the event assignment will be added.
    - event_id: The ID of the event to which the assignment belongs.
    - variable_id: The ID of the parameter or species for which the assignment is defined.
    - formula: The formula representing the assignment.

    Returns:
    - EventAssignment: The created event assignment object.
    """
    event = model.getEvent(event_id)
    if not event:
        raise ValueError(f"Event with ID '{event_id}' not found in the model.")

    assignment: libsbml.EventAssignment = event.createEventAssignment()
    _check(assignment, "create event assignment")
    _check(assignment.setVariable(variable_id), "set event assignment variable")

    # Convert the formula string into an ASTNode and assign it to the event assignment
    math_ast: libsbml.ASTNode = libsbml.parseL3Formula(formula)
    _check(math_ast, "create AST for event assignment formula")
    _check(assignment.setMath(math_ast), "set math on event assignment")

    return assignment


######################
### SBML Validator ###
######################


class validateSBML:
    def __init__(self, ucheck: bool = False):
        self.reader = libsbml.SBMLReader()
        self.ucheck = ucheck
        self.numinvalid = 0

    def validate(self, file: str | Path, quiet: bool = True) -> None:
        if not os.path.exists(file):
            print("[Error] %s : No such file." % file)
            self.numinvalid += 1
            return

        start = time.time()
        sbmlDoc: libsbml.SBMLDocument = libsbml.readSBML(file)
        stop = time.time()
        timeRead = (stop - start) * 1000
        errors = sbmlDoc.getNumErrors()

        seriousErrors = False

        numReadErr = 0
        numReadWarn = 0
        errMsgRead = ""

        if errors > 0:
            for i in range(errors):
                error: libsbml.SBMLError = sbmlDoc.getError(i)
                severity = error.getSeverity()
                if (severity == libsbml.LIBSBML_SEV_ERROR) or (
                    severity == libsbml.LIBSBML_SEV_FATAL
                ):
                    seriousErrors = True
                    numReadErr += 1
                else:
                    numReadWarn += 1

                error_msg: libsbml.SBMLErrorLog = sbmlDoc.getErrorLog()
                errMsgRead = error_msg.toString()

        # If serious errors are encountered while reading an SBML document, it
        # does not make sense to go on and do full consistency checking because
        # the model may be nonsense in the first place.

        numCCErr = 0
        numCCWarn = 0
        errMsgCC = ""
        skipCC = False
        timeCC = 0.0

        if seriousErrors:
            skipCC = True
            errMsgRead += "Further consistency checking and validation aborted."
            self.numinvalid += 1
        else:
            sbmlDoc.setConsistencyChecks(
                libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, self.ucheck
            )
            start = time.time()
            failures = sbmlDoc.checkConsistency()
            stop = time.time()
            timeCC = (stop - start) * 1000

            if failures > 0:

                isinvalid = False
                for i in range(failures):
                    error: libsbml.SBMLError = sbmlDoc.getError(i)
                    severity = error.getSeverity()
                    if (severity == libsbml.LIBSBML_SEV_ERROR) or (
                        severity == libsbml.LIBSBML_SEV_FATAL
                    ):
                        numCCErr += 1
                        isinvalid = True
                    else:
                        numCCWarn += 1

                if isinvalid:
                    self.numinvalid += 1

                error_msg: libsbml.SBMLErrorLog = sbmlDoc.getErrorLog()
                errMsgCC = error_msg.toString()

        if not quiet:
            print("================= SBML Validation summary =================")
            print("                 filename : %s" % file)
            print("         file size (byte) : %d" % (os.path.getsize(file)))
            print("           read time (ms) : %f" % timeRead)

            if not skipCC:
                print("        c-check time (ms) : %f" % timeCC)
            else:
                print("        c-check time (ms) : skipped")

            print("      validation error(s) : %d" % (numReadErr + numCCErr))
            if not skipCC:
                print("    (consistency error(s)): %d" % numCCErr)
            else:
                print("    (consistency error(s)): skipped")

            print("    validation warning(s) : %d" % (numReadWarn + numCCWarn))
            if not skipCC:
                print("  (consistency warning(s)): %d" % numCCWarn)
            else:
                print("  (consistency warning(s)): skipped")

            if errMsgRead or errMsgCC:
                print()
                print("===== validation error/warning messages =====\n")
                if errMsgRead:
                    print(errMsgRead)
                if errMsgCC:
                    print("*** consistency check ***\n")
                    print(errMsgCC)
            print("===========================================================")
