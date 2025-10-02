from typing import Tuple
from pathlib import Path

EX_DIR = Path(__file__).parent.resolve()
BASE_OUTPUT_DIR = EX_DIR / "output"
DATA_DIR = EX_DIR / "data"


def output_dirs(name: str) -> Tuple[Path, Path]:
    dir_path = BASE_OUTPUT_DIR / name

    BASE_OUTPUT_DIR.mkdir(exist_ok=True)
    dir_path.mkdir(exist_ok=True)
    return dir_path, dir_path / "ensemble"
