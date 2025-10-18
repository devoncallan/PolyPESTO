from functools import partial
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pypesto  # type: ignore
from pypesto import Result  # type: ignore
from pypesto.C import (  # type: ignore
    AMICI_STATUS,
    AMICI_T,
    AMICI_X,
    AMICI_Y,
    EnsembleType,
)
from pypesto.ensemble import Ensemble, EnsemblePrediction  # type: ignore
from pypesto.objective import AmiciObjective  # type: ignore
from pypesto.predict import AmiciPredictor  # type: ignore
from pypesto.problem import Problem as PypestoProblem  # type: ignore


def create_ensemble(result: Result) -> Ensemble:

    from petab.v1.parameters import map_unscale, unscale

    prob: PypestoProblem = result.problem
    x_names = prob.get_reduced_vector(prob.x_names)
    x_scales = prob.get_reduced_vector(prob.x_scales)

    ens = Ensemble.from_sample(
        result=result,
        remove_burn_in=True,
        chain_slice=slice(None, None, 1),
        x_names=x_names,
        ensemble_type=EnsembleType.sample,
        lower_bound=np.array(list(map_unscale(prob.lb, x_scales))),
        upper_bound=np.array(list(map_unscale(prob.ub, x_scales))),
    )

    for ix, name in enumerate(ens.x_names):
        ens.x_vectors[ix, :] = unscale(ens.x_vectors[ix, :], x_scales[ix])

    return ens


def create_predictor(prob: PypestoProblem, output_type: str) -> AmiciPredictor:

    obj: AmiciObjective = prob.objective

    if output_type == AMICI_Y:
        output_ids = obj.amici_model.getObservableIds()
    elif output_type == AMICI_X:
        output_ids = obj.amici_model.getStateIds()
    else:
        raise ValueError(f"Unknown output type: {output_type}")

    # print("AMICI MODEL PARAMS in create_predictor:")
    # print(obj.amici_model.getSolver().getSensitivityMethod())
    # print(obj.amici_model.getSolver().getSensitivityOrder())
    # print(obj.amici_model.getSolver().getReturnDataReportingMode())
    # print("==========")

    # print("AMICI MODEL PARAMS in create_predictor (direct):")
    # print(obj.amici_solver.getSensitivityMethod())
    # print(obj.amici_solver.getSensitivityOrder())
    # print(obj.amici_solver.getReturnDataReportingMode())
    # print("==========")

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

    print(f"\n==== Running ensemble predictions ====")

    predictor = create_predictor(pred_prob, output_type)

    engine = pypesto.engine.MultiProcessEngine(**kwargs)
    ensemble_pred = ensemble.predict(
        predictor=predictor,
        prediction_id=output_type,
        engine=engine,
        sensi_orders=(0, 1),
        include_llh_weights=True,
        include_sigmay=True,
    )
    return ensemble_pred


def summarize_ensemble(
    ens: Ensemble, percentiles: Sequence[int] = (5, 25, 75, 95)
) -> pd.DataFrame:
    """
    Create a comprehensive DataFrame combining ensemble statistics and identifiability info.

    Parameters
    ----------
    ens : Ensemble
        PyPESTO Ensemble object containing parameter vectors
    percentiles : Sequence[int], optional
        Percentiles to compute, by default (5, 25, 75, 95)

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per parameter containing ensemble statistics
    """

    from pypesto.C import PERCENTILE
    from petab.v1.C import PARAMETER_ID

    # Compute ensemble summary statistics
    if ens.x_vectors.size == 0:
        print("Ensemble is empty - cannot summarize")
        return pd.DataFrame()  # Return an empty DataFrame if ensemble is empty
    
    summary_dict = ens.compute_summary(percentiles_list=percentiles)

    # Convert summary dict to DataFrame
    # Each key in summary_dict maps to an array with one value per parameter
    summary_data = {}
    for stat_name, values in summary_dict.items():
        col_name = stat_name.replace(" ", "_")
        summary_data[col_name] = values

    summary_df = pd.DataFrame(summary_data, index=ens.x_names)
    summary_df.index.name = PARAMETER_ID

    # Combine the DataFrames
    # id_df already has parameterId, lowerBound, upperBound, ensemble_mean, ensemble_std, ensemble_median
    # We want to add the additional percentile columns from summary_df

    # Select only the percentile columns from summary_df (avoid duplicating mean/std/median)
    percentile_cols = [col for col in summary_df.columns if col.startswith(PERCENTILE)]

    # Combine with identifiability info
    result_df = ens.check_identifiability()
    for col in percentile_cols:
        result_df[col] = summary_df[col]

    return result_df


# from pypesto.objective.amici import AmiciObjectBuilder

# AmiciObjectBuilder.
# from pypesto.petab.objective_creator import AmiciObjectiveCreator

# AmiciObjectiveCreator.prediction_to_petab_measurement_df

def ensemble_summary(ens: Ensemble) -> pd.DataFrame:
    pass
