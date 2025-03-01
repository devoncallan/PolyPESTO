"""
Analysis package for PolyPESTO results.

This package provides functions for analyzing and comparing parameter
estimation results across different experiments and conditions.
"""

from polypesto.analysis.comparison import (
    create_parameter_comparison_df,
    compare_parameters_across_conditions,
)

__all__ = [
    'create_parameter_comparison_df',
    'compare_parameters_across_conditions',
]