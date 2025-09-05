"""
Visualization package for PolyPESTO results.

This package provides functions for visualizing parameter estimation results.
"""

from polypesto.visualization.optimize import plot_waterfall, plot_optimization_scatter
from polypesto.visualization.profile import plot_profiles
from polypesto.visualization.sample import (
    plot_sampling_scatter,
    plot_parameter_traces,
    plot_confidence_intervals,
)
from polypesto.visualization.predict import plot_ensemble_predictions

from polypesto.visualization.measurements import (
    plot_measurements,
    plot_all_measurements,
)

# from polypesto.visualization.compare import (
#     plot_comparisons_1D,
#     plot_all_comparisons_1D,
#     plot_comparisons_2D,
#     plot_all_comparisons_2D,
#     plot_all_comparisons_1D_fill,
# )

# from polypesto.visualization.study import (
#     plot_results,
#     plot_all_results,
#     plot_all_ensemble_predictions,
# )
from polypesto.visualization.results import plot_results

from polypesto.visualization.predict import (
    plot_ensemble_predictions
)

__all__ = [
    "plot_waterfall",
    "plot_optimization_scatter",
    "plot_profiles",
    "plot_sampling_scatter",
    "plot_parameter_traces",
    "plot_confidence_intervals",
    "plot_ensemble_predictions",
    "plot_measurements",
    "plot_all_measurements",
    # "plot_comparisons_1D",
    "plot_all_comparisons_1D",
    "plot_all_results",
    "plot_all_ensemble_predictions",
    "plot_ensemble_predictions",
    "plot_all_comparisons_2D",
    "plot_all_comparisons_1D_fill",
]
