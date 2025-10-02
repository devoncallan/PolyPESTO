from pathlib import Path

from polypesto.utils import filepath
from .types import StudyKey


class StudyPaths:

    def __init__(self, study_dir: str | Path):
        self.study_dir = Path(study_dir)

    @filepath
    def metadata(self) -> Path:
        """Path to study metadata JSON file."""
        return self.study_dir / "metadata.json"

    @filepath
    def true_params(self) -> Path:
        """Path to all study true parameters (ParameterGroup) JSON file."""
        return self.study_dir / "true_params.json"

    def prob_dir(self, key: StudyKey) -> Path:
        return self.study_dir / key.param_id / key.prob_id
