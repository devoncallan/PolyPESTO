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
