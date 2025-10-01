from __future__ import annotations
from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

from ..conditions import SimConditions, create_sim_conditions
from ..params import ParameterSet


def _transpose_study_conditions(
    conds: Dict[str, List[ArrayLike]],
) -> List[Dict[str, np.ndarray]]:
    """Transpose study conditions from `dict of lists` to `list of dicts`.

    conds = {
        "A0": [[0.25, 0.50], [0.50, 0.75], [0.25, 0.75]],
        "B0": [[0.75, 0.50], [0.50, 0.25], [0.75, 0.25]],
    }

    conds_list = [
        {"A0": [0.25, 0.50], "B0": [0.75, 0.50]},
        {"A0": [0.50, 0.75], "B0": [0.50, 0.25]},
        {"A0": [0.25, 0.75], "B0": [0.75, 0.25]},
    ]

    """

    if not conds:
        return []

    conds_list: List[Dict[str, np.ndarray]] = []

    num_exps_dict = {k: len(v) for k, v in conds.items()}
    num_exps_list = list(num_exps_dict.values())
    assert all(
        n == num_exps_list[0] for n in num_exps_list
    ), f"All condition lists must have the same length. Actual lengths: {num_exps_dict}"

    num_exps = num_exps_list[0]
    for i in range(num_exps):
        exp_conds = {k: np.array(v[i]) for k, v in conds.items()}

        assert (
            len(set(len(cond) for cond in exp_conds.values())) == 1
        ), f"All conditions in a single experiment {i} must have the same length. Actual lengths: {[len(cond) for cond in exp_conds.values()]}"

        conds_list.append(exp_conds)

    return conds_list


def create_study_conditions(
    conds: Dict[str, List[ArrayLike]],
    t_evals: ArrayLike | List[ArrayLike],
    noise_levels: float | List[float] = 0.0,
) -> Dict[str, List[SimConditions]]:

    sim_conds: Dict[str, List[SimConditions]] = {}

    conds_list = _transpose_study_conditions(conds)
    prob_ids = [f"prob_{i}" for i in range(len(conds_list))]
    raw_conds_dict = dict(zip(prob_ids, conds_list))

    # Create simulation conditions for each parameter set
    for prob_id, raw_conds in raw_conds_dict.items():

        sim_conds[prob_id] = create_sim_conditions(
            true_params=ParameterSet.empty(),
            conds=raw_conds,
            t_evals=t_evals,
            noise_levels=noise_levels,
        )

    return sim_conds
