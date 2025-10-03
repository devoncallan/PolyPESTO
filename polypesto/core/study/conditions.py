from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np
from numpy.typing import ArrayLike

from polypesto.utils import ID

from ..params import ParameterSet
from ..problem.simulate import SimConditions, create_sim_conditions


def _transpose_study_conditions(
    conds: Mapping[str, List[ArrayLike]],
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
    conds: Mapping[str, List[ArrayLike]],
    t_evals: ArrayLike | List[ArrayLike],
    meas_noise: float | List[float] = 0.0,
) -> Dict[str, List[SimConditions]]:

    sim_conds: Dict[str, List[SimConditions]] = {}

    conds_list = _transpose_study_conditions(conds)
    prob_ids = ID.make_prob_ids(len(conds_list))

    raw_conds_dict = dict(zip(prob_ids, conds_list, strict=True))

    # Create simulation conditions for each parameter set
    for prob_id, raw_conds in raw_conds_dict.items():

        sim_conds[prob_id] = create_sim_conditions(
            conds=raw_conds,
            true_params=ParameterSet.empty(),
            t_evals=t_evals,
            meas_noise=meas_noise,
        )

    return sim_conds
