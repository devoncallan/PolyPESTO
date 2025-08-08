import numpy as np
from pypesto import Result
import pypesto

from pypesto.ensemble import Ensemble, EnsemblePrediction
from pypesto.C import EnsembleType
from pypesto.objective import AmiciObjective
from pypesto.C import AMICI_STATUS, AMICI_T, AMICI_X, AMICI_Y
from pypesto.predict import AmiciPredictor

from polypesto.core.experiment import Experiment


def create_ensemble(exp: Experiment, result: Result) -> Ensemble:

    problem = exp.pypesto_problem
    x_names = problem.get_reduced_vector(problem.x_names)

    ensemble = Ensemble.from_sample(
        result=result,
        remove_burn_in=True,
        chain_slice=slice(None, None, 10),
        x_names=x_names,
        ensemble_type=EnsembleType.sample,
        lower_bound=result.problem.lb,
        upper_bound=result.problem.ub,
    )

    return ensemble


def create_predictor(exp: Experiment, output_type: str) -> AmiciPredictor:

    obj: AmiciObjective = exp.pypesto_problem.objective

    if output_type == AMICI_Y:
        output_ids = obj.amici_model.getObservableIds()
    elif output_type == AMICI_X:
        output_ids = obj.amici_model.getStateIds()
    else:
        raise ValueError(f"Unknown output type: {output_type}")

    # This post_processor will transform the output of the simulation tool
    # such that the output is compatible with the next steps.
    def post_processor(amici_outputs, output_type, output_ids):
        outputs = [
            (
                amici_output[output_type]
                if amici_output[AMICI_STATUS] == 0
                else np.full((len(amici_output[AMICI_T]), len(output_ids)), np.nan)
            )
            for amici_output in amici_outputs
        ]
        return outputs

    from functools import partial

    post_processor_bound = partial(
        post_processor,
        output_type=output_type,
        output_ids=output_ids,
    )

    predictor = AmiciPredictor(
        amici_objective=obj,
        post_processor=post_processor_bound,
        output_ids=output_ids,
    )

    return predictor


def predict_with_ensemble(
    ensemble: Ensemble,
    test_exp: Experiment,
    output_type: str = AMICI_Y,
    **kwargs,
) -> EnsemblePrediction:

    predictor = create_predictor(test_exp, output_type)

    engine = pypesto.engine.MultiProcessEngine(**kwargs)
    ensemble_pred = ensemble.predict(
        predictor=predictor,
        prediction_id=output_type,
        engine=engine,
    )
    return ensemble_pred
