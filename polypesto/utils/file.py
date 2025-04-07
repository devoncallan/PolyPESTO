import os
import json
from typing import Any, Dict, TypeAlias

Filename: TypeAlias = str
Filepath: TypeAlias = str
Directory: TypeAlias = str


def read_json(filepath: Filepath, **kwargs) -> dict:
    with open(filepath, "r", **kwargs) as file:
        data = json.load(file)
    return data


def write_json(filepath: Filepath, data: dict, **kwargs) -> None:
    with open(filepath, "w", **kwargs) as file:
        json.dump(data, file, indent=4)


def pickle_results(result, filepath: str = "results.pkl"):

    # Check that filepath extension is pkl
    if not filepath.endswith(".pkl"):
        raise ValueError(f"Filepath ({filepath}) must end with .pkl")

    import numpy as np
    import pickle

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [deep_convert(v) for v in obj]
        elif isinstance(obj, (np.generic,)):
            return obj.item()
        else:
            return obj

    clean_data = deep_convert(result)

    with open(filepath, "wb") as f:
        pickle.dump(clean_data, f)
