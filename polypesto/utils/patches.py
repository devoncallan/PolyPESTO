from pathlib import Path


def _safe_log(msg: str):
    try:
        import warnings

        warnings.warn(f"[polypesto patches] {msg}")
    except Exception:
        print(f"[polypesto patches] {msg}")
        pass


def patch_petab_from_yaml() -> None:
    """Adding base_path support to petab.Problem.from_yaml for absolute file paths."""

    try:
        from pathlib import Path
        from petab.v1 import Problem

        # https://github.com/PEtab-dev/libpetab-python/blob/9a4efb46f91f0af06f9e857ab1656f103281fbbf/petab/v1/problem.py#L254
        @staticmethod
        def _from_yaml(
            yaml_config: dict | Path | str, base_path: str | Path = None
        ) -> Problem:
            """
            Factory method to load model and tables as specified by YAML file.

            Arguments:
                yaml_config: PEtab configuration as dictionary or YAML file name
                base_path: Base directory or URL to resolve relative paths
            """

            from petab.v1 import (
                yaml,
                parameters,
                core,
                measurements,
                conditions,
                observables,
                mapping,
                format_version,
            )
            from petab.v1.models import MODEL_TYPE_SBML
            from petab.v1.models.model import model_factory
            from petab.v1.C import (
                FORMAT_VERSION,
                MODEL_FILES,
                MODEL_LOCATION,
                MODEL_LANGUAGE,
                EXTENSIONS,
                MAPPING_FILES,
                PARAMETER_FILE,
                CONDITION_FILES,
                MEASUREMENT_FILES,
                OBSERVABLE_FILES,
                VISUALIZATION_FILES,
                SBML_FILES,
            )
            from petab.v1.yaml import get_path_prefix
            from warnings import warn

            if isinstance(yaml_config, Path):
                yaml_config = str(yaml_config)

            if isinstance(yaml_config, str):
                if base_path is None:
                    base_path = get_path_prefix(yaml_config)
                yaml_config = yaml.load_yaml(yaml_config)

            def get_path(filename):
                if base_path is None:
                    return filename
                return f"{base_path}/{filename}"

            if yaml.is_composite_problem(yaml_config):
                raise ValueError(
                    "petab.Problem.from_yaml() can only be used for "
                    "yaml files comprising a single model. "
                    "Consider using "
                    "petab.CompositeProblem.from_yaml() instead."
                )

            if yaml_config[FORMAT_VERSION] not in {"1", 1, "1.0.0", "2.0.0"}:
                raise ValueError(
                    "Provided PEtab files are of unsupported version "
                    f"{yaml_config[FORMAT_VERSION]}. Expected "
                    f"{format_version.__format_version__}."
                )
            if yaml_config[FORMAT_VERSION] == "2.0.0":
                warn("Support for PEtab2.0 is experimental!", stacklevel=2)
                warn(
                    "Using petab.v1.Problem with PEtab2.0 is deprecated. "
                    "Use petab.v2.Problem instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            problem0 = yaml_config["problems"][0]

            if isinstance(yaml_config[PARAMETER_FILE], list):
                parameter_df = parameters.get_parameter_df(
                    [get_path(f) for f in yaml_config[PARAMETER_FILE]]
                )
            else:
                parameter_df = (
                    parameters.get_parameter_df(get_path(yaml_config[PARAMETER_FILE]))
                    if yaml_config[PARAMETER_FILE]
                    else None
                )

            if yaml_config[FORMAT_VERSION] in [1, "1", "1.0.0"]:
                if len(problem0[SBML_FILES]) > 1:
                    # TODO https://github.com/PEtab-dev/libpetab-python/issues/6
                    raise NotImplementedError(
                        "Support for multiple models is not yet implemented."
                    )

                model = (
                    model_factory(
                        get_path(problem0[SBML_FILES][0]),
                        MODEL_TYPE_SBML,
                        model_id=None,
                    )
                    if problem0[SBML_FILES]
                    else None
                )
            else:
                if len(problem0[MODEL_FILES]) > 1:
                    # TODO https://github.com/PEtab-dev/libpetab-python/issues/6
                    raise NotImplementedError(
                        "Support for multiple models is not yet implemented."
                    )
                if not problem0[MODEL_FILES]:
                    model = None
                else:
                    model_id, model_info = next(iter(problem0[MODEL_FILES].items()))
                    model = model_factory(
                        get_path(model_info[MODEL_LOCATION]),
                        model_info[MODEL_LANGUAGE],
                        model_id=model_id,
                    )

            measurement_files = [
                get_path(f) for f in problem0.get(MEASUREMENT_FILES, [])
            ]
            # If there are multiple tables, we will merge them
            measurement_df = (
                core.concat_tables(measurement_files, measurements.get_measurement_df)
                if measurement_files
                else None
            )

            condition_files = [get_path(f) for f in problem0.get(CONDITION_FILES, [])]
            # If there are multiple tables, we will merge them
            condition_df = (
                core.concat_tables(condition_files, conditions.get_condition_df)
                if condition_files
                else None
            )

            visualization_files = [
                get_path(f) for f in problem0.get(VISUALIZATION_FILES, [])
            ]
            # If there are multiple tables, we will merge them
            visualization_df = (
                core.concat_tables(visualization_files, core.get_visualization_df)
                if visualization_files
                else None
            )

            observable_files = [get_path(f) for f in problem0.get(OBSERVABLE_FILES, [])]
            # If there are multiple tables, we will merge them
            observable_df = (
                core.concat_tables(observable_files, observables.get_observable_df)
                if observable_files
                else None
            )

            mapping_files = [get_path(f) for f in problem0.get(MAPPING_FILES, [])]
            # If there are multiple tables, we will merge them
            mapping_df = (
                core.concat_tables(mapping_files, mapping.get_mapping_df)
                if mapping_files
                else None
            )

            return Problem(
                condition_df=condition_df,
                measurement_df=measurement_df,
                parameter_df=parameter_df,
                observable_df=observable_df,
                model=model,
                visualization_df=visualization_df,
                mapping_df=mapping_df,
                extensions_config=yaml_config.get(EXTENSIONS, {}),
            )

        Problem.from_yaml = staticmethod(_from_yaml)

    except Exception as e:
        _safe_log(f"petab from_yaml patch skipped: {e!r}")


def patch_pypesto_importer_from_yaml() -> None:
    try:
        from petab.v1 import Problem
        from pypesto.petab.importer import PetabImporter
        from pypesto.C import AMICI

        _orig = getattr(PetabImporter, "from_yaml")

        @staticmethod
        def _from_yaml(
            yaml_config: dict | str,
            output_folder: str = None,
            model_name: str = None,
            simulator_type: str = AMICI,
            base_path: str | Path = None,
        ) -> PetabImporter:
            """Simplified constructor using a petab yaml file."""
            petab_problem = Problem.from_yaml(yaml_config, base_path=base_path)

            return PetabImporter(
                petab_problem=petab_problem,
                output_folder=output_folder,
                model_name=model_name,
                simulator_type=simulator_type,
            )

        PetabImporter.from_yaml = staticmethod(_from_yaml)

    except Exception as e:
        _safe_log(f"pypesto PetabImporter.from_yaml patch skipped: {e!r}")


def apply_patches() -> None:
    import warnings

    try:
        patch_petab_from_yaml()
        patch_pypesto_importer_from_yaml()
    except Exception as e:
        _safe_log(f"Applying patches failed: {e!r}")

        warnings.warn(f"[polypesto] patch bootstrap failed: {e!r}")


def get_patched_petab_problem():
    """Get Problem class with patches applied. Thread-safe and runs once per process."""
    import functools
    
    @functools.lru_cache(maxsize=1)
    def _get_patched_problem():
        from petab.v1 import Problem

        # Check if already patched
        if hasattr(Problem.from_yaml, "__wrapped__") or "base_path" in str(
            Problem.from_yaml
        ):
            return Problem

        # Apply patches
        try:
            apply_patches()
        except Exception:
            pass  # Graceful fallback

        return Problem

    return _get_patched_problem()
