import functools
from typing import Callable, Tuple, Any, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pypesto.result import Result


plot_func: TypeAlias = Callable[[Result, Any], Tuple[Figure, Any]]


def safe_plot(func: plot_func) -> plot_func:
    """
    Decorator that catches any exceptions in plotting functions
    and returns an empty figure and axes if an error occurs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Figure, Any]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return plt.subplots()

    return wrapper
