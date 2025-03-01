# PolyPESTO Guide

## Build/Run Commands
- Run an experiment: `python experiments/[experiment_type]/[experiment_name]/exp.py`
- Run analysis: Open and run Jupyter notebooks in experiment directories
- Build Docker environment: `docker build -f docker/Dockerfile -t polypesto .`
- Run in container: `docker run -it -v $(pwd):/PolyPESTO polypesto`
- Generate SBML model: Use `Model.sbml_model_def()` from any model class

## Code Style
- **Imports**: stdlib first, third-party second, project modules last
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Types**: Use type annotations extensively; import from typing module
- **Docstrings**: Google style with Args/Returns sections
- **Error handling**: Assertions for validation, raise ValueError for invalid inputs
- **Classes**: Use Protocol for interfaces, @dataclass for data containers
- **Testing**: Add experimental hypotheses as docstrings at module level
- **Formatting**: Follow PEP 8 guidelines
- **Organization**: Keep model definitions separate from experiment code