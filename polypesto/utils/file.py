from typing import Callable, Any, Dict, TypeVar
from functools import wraps
from pathlib import Path
import json
import os


def filepath(func: Callable[..., Path]) -> property:
    """
    Decorator that handles lazy directory creation.

    Creates a property that returns a Path, ensuring the parent
    directory exists when the path is accessed.

    """

    @wraps(func)
    def wrapper(self) -> Path:
        path = func(self)
        if not isinstance(path, Path):
            path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        return path

    wrapper.__annotations__["return"] = Path
    return property(wrapper)


def read_json(filepath: str | Path, **kwargs) -> Dict[Any, Any]:
    """Read a JSON file and return its contents.

    Args:
        filepath (str | Path): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", **kwargs) as file:
        data = json.load(file)

    return data


def write_json(filepath: str | Path, data: dict, **kwargs) -> None:
    """Write a dictionary to a JSON file.

    Args:
        filepath (str | Path): The path to the JSON file.
        data (dict): The data to write to the JSON file.
    """
    filepath = Path(filepath)
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w", **kwargs) as file:
        json.dump(data, file, indent=4)
