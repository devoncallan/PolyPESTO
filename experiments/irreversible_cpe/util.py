import numpy as np

from polypesto.core.experiment import create_simulation_conditions
from polypesto.core.experiment import Experiment
from polypesto.core.study import Study, create_study

from polypesto.models.CRP2 import IrreversibleCPE


def get_test_study(study: Study, test_dir: str) -> Study:

    t_eval = np.arange(0, 1, 0.1)
    test_conditions = create_simulation_conditions(
        dict(
            name=["Test_Study"],
            t_eval=[t_eval],
            conditions=dict(
                fA0=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], cM0=[[1.0] * 9]
            ),
            fit_params=[None],
            noise_level=[0.00],
        )
    )

    test_study = create_study(
        model=IrreversibleCPE,
        simulation_params=study.simulation_params,
        conditions=test_conditions,
        base_dir=test_dir,
        overwrite=False,
    )

    return test_study
