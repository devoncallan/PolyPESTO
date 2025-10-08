import functools
from pathlib import Path
from typing import Callable, Optional, Tuple, Any, TypeAlias

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


from contextlib import contextmanager


@contextmanager
def save_plot(
    path: Optional[str | Path] = None,
    overwrite: bool = False,
    dpi: int = 300,
    close: bool = True,
    safe: bool = True,
):
    """Context manager for creating and saving a plot."""

    try:
        yield
        if path:
            path = Path(path)
            if overwrite or not path.exists():
                plt.gcf().savefig(path, dpi=dpi)
    except Exception as e:
        if safe:
            print(f"Error saving plot to {path}: {e}")
            plt.figure()
        else:
            raise e
    finally:
        if close:
            plt.close()
