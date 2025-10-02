from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from petab.v1.parameters import scale  # type: ignore

from polypesto.core import Study, calculate_cis

"""
x axis value: (fA0)
y axis value: parameter confidence interval (e.g. 5th to 95th percentile)

x_values = (0.1, 0.3, 0.5, 0.7, 0.9)
param_value = list(problems)

"""

# idx = free_param_names.index(param_name)
#             param_value = scale(param_value, problem.x_scales[idx])

# problem_ids ...
# conditions ...
# 

def plot_comparisons_1D(
    study: Study, param_id: str, axes: Optional[List[Axes]]
) -> List[Axes]:

    # Plot confidence intervals for all parameters across a certain condition

    problems = study.get_problems(param_id=param_id)
    results = study.get_results(param_id=param_id)

    example_problem = list(problems.values())[0]
    param_names = example_problem.petab_problem.get_optimization_parameters()
    
    example_problem.sim_conditions[0].conds.to_dict()

    true_params = study.true_params[param_id]

    if axes is None:
        fig, axes = plt.subplots(1, len(true_params), figsize=(4 * len(true_params), 4))

    for i, key in enumerate(problems.items()):

        result = results[key]
        true_params_dict = true_params.to_dict()
        cis = calculate_cis(result)

        for j, param_name in enumerate(param_names):

            lb, med, ub = cis[param_name]

            lb = np.abs(med - lb)
            ub = np.abs(ub - med)

            axes[j].errorbar(
                i,  # conditions
                med,
                yerr=[[lb], [ub]],
                fmt="o",
                color=colors[j],
                capsize=5,
                label=f"{param_id}" if i == 0 else None,
            )

            if i == 0:
                plot_params = true_params_dict[param_name]
                axes[j].axhline(plot_params, color="red", linestyle="--")
                axes[j].set_ylim([-2, 2])

        pass

    pass
