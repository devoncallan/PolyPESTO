from typing import List, Optional
from matplotlib.axes import Axes
from polypesto.core.study import Study


"""
x axis value: (fA0)
y axis value: parameter confidence interval (e.g. 5th to 95th percentile)

x_values = (0.1, 0.3, 0.5, 0.7, 0.9)
param_value = list(problems)

"""



def plot_comparisons_1D(
    study: Study, param_id: str, axes: Optional[List[Axes]]
) -> List[Axes]:
    
    # Plot confidence intervals for all parameters across a certain condition
    
    
    pass
