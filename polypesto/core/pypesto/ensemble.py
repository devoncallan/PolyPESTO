from typing import Dict, List, Sequence
from functools import partial

import numpy as np

import pypesto
from pypesto import Result
from pypesto.ensemble import Ensemble, EnsemblePrediction
from pypesto.objective import AmiciObjective
from pypesto.C import EnsembleType, AMICI_STATUS, AMICI_T, AMICI_X, AMICI_Y
from pypesto.predict import AmiciPredictor
from pypesto.problem import Problem as PypestoProblem


def create_ensemble(prob: PypestoProblem, result: Result) -> Ensemble:

    x_names = prob.get_reduced_vector(prob.x_names)

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


def create_predictor(prob: PypestoProblem, output_type: str) -> AmiciPredictor:

    obj: AmiciObjective = prob.objective

    if output_type == AMICI_Y:
        output_ids = obj.amici_model.getObservableIds()
    elif output_type == AMICI_X:
        output_ids = obj.amici_model.getStateIds()
    else:
        raise ValueError(f"Unknown output type: {output_type}")

    # This post_processor will transform the output of the simulation tool
    # such that the output is compatible with the next steps.
    def post_processor(
        amici_outputs: List[Dict[str, np.ndarray]],
        _output_type: str,
        _output_ids: Sequence[str],
    ) -> List[np.ndarray]:
        outputs = [
            (
                amici_output[_output_type]
                if amici_output[AMICI_STATUS] == 0
                else np.full((len(amici_output[AMICI_T]), len(_output_ids)), np.nan)
            )
            for amici_output in amici_outputs
        ]
        return outputs

    post_processor_bound = partial(
        post_processor,
        _output_type=output_type,
        _output_ids=output_ids,
    )

    predictor = AmiciPredictor(
        amici_objective=obj,
        post_processor=post_processor_bound,
        output_ids=output_ids,
    )

    return predictor


def predict_with_ensemble(
    ensemble: Ensemble,
    pred_prob: PypestoProblem,
    output_type: str = AMICI_Y,
    **kwargs,
) -> EnsemblePrediction:

    predictor = create_predictor(pred_prob, output_type)

    engine = pypesto.engine.MultiProcessEngine(**kwargs)
    ensemble_pred = ensemble.predict(
        predictor=predictor,
        prediction_id=output_type,
        engine=engine,
    )
    return ensemble_pred
