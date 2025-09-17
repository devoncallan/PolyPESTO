from typing import List

import numpy as np


from polypesto.core.experiment import Experiment
from .irreversible_cpe import BinaryIrreversible
from .reversible_cpe import BinaryReversible


def modify_experiments(experiments: List[Experiment]) -> List[Experiment]:

    # Convert tkey in data to conversion
    # Add fA and fB to obs

    new_exps = []
    for exp in experiments:

        datasets = exp.data
        cond = exp.conds.values.to_dict()

        A0 = B0 = None
        if "A0" in cond and "B0" in cond:
            A0 = cond["A0"]
            B0 = cond["B0"]
        elif "fA0" in cond and "cM0" in cond:
            fA0 = cond["fA0"]
            cM0 = cond["cM0"]
            A0 = fA0 * cM0
            B0 = (1 - fA0) * cM0
        else:
            raise ValueError(
                f'Conditions must include either ("A0", "B0") or ("fA0", "cM0"). Actual: {list(cond.keys())}'
            )

        assert A0 is not None and B0 is not None
        assert A0 != 0 and B0 != 0

        fA0 = A0 / (A0 + B0)
        fB0 = B0 / (A0 + B0)

        new_datasets = []
        for ds in datasets:

            assert ds.tkey in ds.data.columns
            assert "xA" in ds.obs_map and "xB" in ds.obs_map

            new_data = ds.data.copy()
            new_data[ds.tkey] = (
                fA0 * ds.data[ds.obs_map["xA"]] + fB0 * ds.data[ds.obs_map["xB"]]
            )

            mon_A = fA0 * (1 - new_data[ds.obs_map["xA"]])
            mon_B = fB0 * (1 - new_data[ds.obs_map["xB"]])
            new_data["fA"] = mon_A / (mon_A + mon_B)
            new_data["fB"] = mon_B / (mon_A + mon_B)

            ds.obs_map["fA"] = "fA"
            ds.obs_map["fB"] = "fB"

            ds.data = new_data

            new_datasets.append(ds)

        exp = Experiment(
            id=exp.id,
            conds=exp.conds,
            data=new_datasets,
        )
        new_exps.append(exp)

    return new_exps


def create_ensemble_pred_problem(
    data_dir: str, model: BinaryIrreversible | BinaryReversible
):

    from polypesto.core.problem.simulate import write_empty_problem
    from polypesto.core.conditions import create_sim_conditions

    fA0s = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    cM0s = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    A0s = fA0s * cM0s
    B0s = (1 - fA0s) * cM0s

    problem, _ = write_empty_problem(
        prob_dir=data_dir,
        model=model,
        conds=create_sim_conditions(
            true_params={},
            t_evals=np.arange(0.01, 0.9, 0.01),
            conds=dict(
                A0=A0s,
                B0=B0s,
            ),
        ),
    )

    return problem
