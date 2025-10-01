# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolyPESTO is a parameter estimation package for polymerization data using PyPESTO. It provides a framework for modeling chemical kinetics, particularly polymerization reactions, and estimating parameters from experimental data.

## Development Setup

### Installation
The project uses setuptools with pyproject.toml configuration. Install in development mode:
```bash
pip install -e .
```

### Dependencies
- Core dependencies include pypesto[amici,petab,fides], python-libsbml, h5py
- Development dependencies include pytest, pytest-cov for testing
- Visualization tools: plotly, matplotlib, seaborn
- Jupyter support included for notebook development

## Testing

### Running Tests
```bash
pytest
```

### Test Structure
- Tests are located in the `tests/` directory
- Main test fixtures are defined in `tests/conftest.py`
- Test models include "BinaryIrreversible" and "LotkaVolterra"
- Tests use parameterized fixtures for different model types

### Single Test Execution
```bash
pytest tests/path/to/specific_test.py
pytest -k "test_name_pattern"
```

## Architecture

### Core Components

**polypesto.core**: Central framework components
- `Study`: Main class for managing parameter estimation studies across multiple problems and parameter sets
- `Problem`: Individual optimization problems with model, data, and parameter definitions
- `experiment.py`: Experimental data handling
- `conditions.py`: Simulation condition management
- `params.py`: Parameter group and set definitions
- `petab.py`: Integration with PEtab standard
- `pypesto/`: PyPESTO integration utilities

**polypesto.models**: Model definitions
- `ModelBase`: Abstract base class for all models
- `binary/`: Binary polymerization models
- `example/`: Example models like Lotka-Volterra
- `sbml.py`: SBML model utilities
- Models automatically generate AMICI models in `amici_models/` directory

**polypesto.visualization**: Plotting and visualization utilities
- Modular visualization components for different analysis types
- Integration with plotly, matplotlib, and seaborn

**polypesto.utils**: Utility functions
- `file.py`: JSON I/O utilities
- `logging.py`: Logging configuration
- `patches.py`: Compatibility patches

### Key Design Patterns

- Models inherit from `ModelBase` and define SBML representations
- Study objects manage collections of problems with different parameter sets
- Type aliases used extensively for complex nested dictionaries (e.g., `ProbParamDict`)
- AMICI integration for efficient simulation and sensitivity analysis

### Data Flow

1. Models define chemical kinetics via SBML
2. Study objects coordinate parameter estimation across multiple problems
3. Problems contain model + data + parameter bounds
4. PyPESTO performs optimization
5. Results stored and visualized through dedicated modules

## Common Tasks

### Adding New Models
1. Create new model class inheriting from `ModelBase` in `polypesto/models/`
2. Implement required abstract methods for SBML definition
3. Add observables and parameter definitions
4. Test with existing test framework fixtures

### Running Parameter Estimation
Check `polypesto/examples/` for example scripts showing typical workflows.