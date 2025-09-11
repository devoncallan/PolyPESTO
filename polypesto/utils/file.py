import os
import json
from pathlib import Path


def read_json(filepath: str | Path, **kwargs) -> dict:
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
