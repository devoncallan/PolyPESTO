import functools
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def safe_plot(func: Callable) -> Callable:
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
