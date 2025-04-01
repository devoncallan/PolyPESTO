import os
from pathlib import Path
from typing import Dict, List, Tuple

from polypesto.core.experiment import ExperimentPaths


def find_experiment_paths(base_dir: str) -> Dict[Tuple[str, str], ExperimentPaths]:
    """Find all petab.yaml files and map them to their parameter set IDs"""
    base_dir = Path(base_dir)

    experiment_paths = {}
    for yaml_path in sorted(base_dir.glob("**/petab.yaml")):

        paths = ExperimentPaths.from_yaml(yaml_path)
        experiment_paths[paths.get_base_name(), paths.get_exp_id()] = paths

    return experiment_paths
