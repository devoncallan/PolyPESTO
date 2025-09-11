from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from . import petab as pet
from .params import ParameterSet


@dataclass
class Conditions:
    """Initial conditions for a given experiment."""

    exp_id: str
    values: ParameterSet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exp_id": self.exp_id,
            "values": self.values.to_dict(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Conditions":
        return Conditions(
            exp_id=d["exp_id"],
            values=ParameterSet.lazy_from_dict(d["values"]),
        )


@dataclass
class SimConditions(Conditions):
    """Simulation conditions for a given experiment."""

    true_params: ParameterSet
    t_eval: ArrayLike
    noise_level: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "true_params": self.true_params.to_dict(),
                "t_eval": list(self.t_eval),
                "noise_level": self.noise_level,
            }
        )
        return base_dict

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SimConditions":
        return SimConditions(
            exp_id=d["exp_id"],
            values=ParameterSet.lazy_from_dict(d["values"]),
            true_params=ParameterSet.lazy_from_dict(d["true_params"]),
            t_eval=np.array(d["t_eval"]),
            noise_level=d.get("noise_level", 0.0),
        )


def create_conditions(
    conds: Dict[str, ArrayLike], exp_ids: Optional[List[str]] = None
) -> List[Conditions]:
    """Create a list of Conditions from a dictionary of condition values.

    Args:
        conds (Dict[str, ArrayLike]): Dictionary of condition values.
            e.g., `conds = {"A0": [0.25, 0.6], "B0": [0.75, 0.4]}`
        exp_ids (Optional[List[str]]): List of experiment IDs. Defaults to None.
            e.g., `exp_ids = ["c_0", "c_1"]`. If None, will be auto-generated.

    Returns:
        List[Conditions]: List of Conditions objects.
    """

    cond_labels = list(conds.keys())
    len_conds = {k: len(v) for k, v in conds.items()}
    n_conds = len_conds[cond_labels[0]]

    if not np.all(np.array(list(len_conds.values())) == n_conds):
        raise ValueError(
            f"All condition lists must have the same length. Lengths: {len_conds}"
        )

    if exp_ids is None:
        exp_ids = [f"c_{i}" for i in range(n_conds)]
    elif len(exp_ids) != n_conds:
        raise ValueError(
            f"Length of exp_ids ({len(exp_ids)}) must match number of conditions ({n_conds})."
        )

    conditions = []
    for i in range(n_conds):
        cond = Conditions(
            exp_id=exp_ids[i],
            values=ParameterSet.lazy_from_dict(
                {cond_id: conds[cond_id][i] for cond_id in cond_labels}, id=exp_ids[i]
            ),
        )
        conditions.append(cond)

    return conditions


def create_sim_conditions(
    true_params: ParameterSet | Dict[str, float],
    conds: Dict[str, ArrayLike],
    t_evals: ArrayLike | List[ArrayLike],
    noise_levels: float | List[float] = 0.0,
    exp_ids: Optional[List[str]] = None,
) -> List[SimConditions]:
    """Create a list of SimConditions from the provided parameters.

    Args:
        true_params (ParameterSet | Dict[str, float]): True parameter values.
            e.g., `true_params = {"rA": 0.25, "rB": 0.75}`
        conds (Dict[str, ArrayLike]): Dictionary of condition values.
            e.g., `conds = {"A0": [0.25, 0.6], "B0": [0.75, 0.4]}`
        t_evals (ArrayLike | List[ArrayLike]): Time evaluation points.
            e.g., `t_evals = np.linspace(0, 10, 100)` or `t_evals = [np.linspace(0, 10, 100), np.linspace(0, 5, 50)]`
        noise_levels (float | List[float]): Noise levels for the simulations. Defaults to 0.0.
            e.g., `noise_levels = 0.1` or `noise_levels = [0.1, 0.2]`
        exp_ids (Optional[List[str]]): List of experiment IDs. Defaults to None.
            e.g., `exp_ids = ["c_0", "c_1"]`. If None, will be auto-generated.

    Returns:
        List[SimConditions]: List of simulation conditions.
    """

    if isinstance(true_params, dict):
        true_params = ParameterSet.lazy_from_dict(true_params)
    elif not isinstance(true_params, ParameterSet):
        raise ValueError("true_params must be a ParameterSet or a dict.")

    conditions = create_conditions(conds, exp_ids)
    n_conds = len(conditions)

    if isinstance(t_evals, np.ndarray):
        t_evals = [t_evals] * n_conds
    elif len(t_evals) != n_conds:
        raise ValueError(
            f"Length of t_evals ({len(t_evals)}) must match number of conditions ({n_conds})."
        )
    assert isinstance(t_evals, list) and len(t_evals) == n_conds

    if isinstance(noise_levels, float):
        noise_levels = [noise_levels] * n_conds
    elif len(noise_levels) != n_conds:
        raise ValueError(
            f"Length of noise_levels ({len(noise_levels)}) must match number of conditions ({n_conds})."
        )
    assert isinstance(noise_levels, list) and len(noise_levels) == n_conds

    sim_conditions = []
    for i in range(n_conds):

        sim_cond = SimConditions(
            exp_id=conditions[i].exp_id,
            values=conditions[i].values,
            true_params=true_params,
            t_eval=t_evals[i],
            noise_level=noise_levels[i],
        )
        sim_conditions.append(sim_cond)

    return sim_conditions


def conditions_to_df(conds: List[Conditions]) -> pd.DataFrame:
    """Convert a list of Conditions to a PEtab conditions dataframe.

    Args:
        conds (List[Conditions]): List of Conditions objects.

    Returns:
        pd.DataFrame: PEtab conditions dataframe.
    """

    return pet.define_conditions(
        [cond.values.to_dict() for cond in conds],
        exp_ids=[cond.exp_id for cond in conds],
    )
