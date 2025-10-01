from pathlib import Path

from polypesto.utils.file import filepath


class StudyPaths:

    def __init__(self, study_dir: str | Path):
        self.study_dir = Path(study_dir)

    @filepath
    def metadata(self) -> Path:
        return self.study_dir / "metadata.json"

    @filepath
    def true_params(self) -> Path:
        return self.study_dir / "true_params.json"

    @filepath
    def sim_params(self) -> Path:
        return self.study_dir / "sim_params.json"
