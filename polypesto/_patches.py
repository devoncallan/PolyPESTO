# polypesto/_patches.py
from __future__ import annotations

def _safe_log(msg: str):
    try:
        import warnings
        warnings.warn(f"[polypesto patches] {msg}")
    except Exception:
        pass

def _patch_petab_from_yaml():
    try:
        from pathlib import Path
        from typing import Union
        import petab
        from petab.v1.problem import Problem # type: ignore

        _orig = Problem.from_yaml
        def _from_yaml(
            yaml_config: dict | Path | str, base_path: str | Path = None
            ) -> Problem:
            """
            Factory method to load model and tables as specified by YAML file.

            Arguments:
                yaml_config: PEtab configuration as dictionary or YAML file name
                base_path: Base directory or URL to resolve relative paths
            """

            from petab.v1 import yaml, parameters, core, measurements, conditions, observables, mapping, format_version
            from petab.versions import get_major_version
            from petab.v1.models import MODEL_TYPE_SBML
            from petab.v1.models.model import model_factory
            from petab.v1.C import FORMAT_VERSION, MODEL_FILES, MODEL_LOCATION, MODEL_LANGUAGE, EXTENSIONS, MAPPING_FILES, PARAMETER_FILE, CONDITION_FILES, MEASUREMENT_FILES, OBSERVABLE_FILES, VISUALIZATION_FILES, SBML_FILES
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
                    parameters.get_parameter_df(
                        get_path(yaml_config[PARAMETER_FILE])
                    )
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
                    model_id, model_info = next(
                        iter(problem0[MODEL_FILES].items())
                    )
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
                core.concat_tables(
                    measurement_files, measurements.get_measurement_df
                )
                if measurement_files
                else None
            )

            condition_files = [
                get_path(f) for f in problem0.get(CONDITION_FILES, [])
            ]
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

            observable_files = [
                get_path(f) for f in problem0.get(OBSERVABLE_FILES, [])
            ]
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
        
        # install patch
        Problem.from_yaml = staticmethod(_from_yaml)  # type: ignore[attr-defined]
    except Exception as e:
        _safe_log(f"petab from_yaml patch skipped: {e!r}")

def _patch_pypesto_importer_from_yaml():
    try:
        from pathlib import Path
        from typing import Union
        import petab
        from pypesto.petab.importer import PetabImporter  # type: ignore

        _orig = getattr(PetabImporter, "from_yaml")

        def _from_yaml(
            yaml_config: Union[dict, Path, str],
            output_folder: str | None = None,
            model_name: str | None = None,
            simulator_type: str = "amici",
            base_path: Union[str, Path, None] = None,
        ) -> "PetabImporter":
            """
            patched: add base_path and forward to petab.Problem.from_yaml(yaml_config, base_path)
            mirrors your pypesto_import_changes.patch.
            """
            try:
                petab_problem = petab.Problem.from_yaml(yaml_config, base_path)  # type: ignore[arg-type]
            except TypeError:
                # if our petab patch didn't land, fall back to original behavior
                petab_problem = petab.Problem.from_yaml(yaml_config)
            return PetabImporter(
                petab_problem=petab_problem,
                output_folder=output_folder,
                model_name=model_name,
                simulator_type=simulator_type,
            )

        PetabImporter.from_yaml = staticmethod(_from_yaml)  # type: ignore[attr-defined]
    except Exception as e:
        _safe_log(f"pypesto PetabImporter.from_yaml patch skipped: {e!r}")

def _patch_pypesto_sampling_bounds():
    """
    find the sampling function with docstring 'Plot MCMC-based parameter credibility intervals.'
    and wrap it to add show_bounds: bool=False; if true, set xlim to [min(lb), max(ub)].
    mirrors your sampling_bounds.patch without requiring exact function name.
    """
    try:
        import inspect
        import matplotlib
        import pypesto.visualize.sampling as sampling  # type: ignore

        target = None
        for name, obj in vars(sampling).items():
            if callable(obj) and inspect.isfunction(obj) and obj.__doc__:
                if "Plot MCMC-based parameter credibility intervals." in obj.__doc__:
                    target = obj
                    break

        if target is None:
            _safe_log("could not locate the credibility-intervals plotter; bounds toggle not applied.")
            return

        original = target

        def _wrapped(*args, show_bounds: bool = False, **kwargs):
            ax = kwargs.get("ax", None)
            out_ax = original(*args, **kwargs)
            try:
                result = kwargs.get("result", None)
                if result is None and len(args) >= 1:
                    result = args[0]
                if show_bounds and hasattr(result, "problem"):
                    lb = getattr(result.problem, "lb", None)
                    ub = getattr(result.problem, "ub", None)
                    if lb is not None and ub is not None:
                        from math import inf
                        _min = min([v for v in lb if v is not None and v != -inf])
                        _max = max([v for v in ub if v is not None and v != inf])
                        (ax or out_ax).set_xlim(_min, _max)
            except Exception as e:
                _safe_log(f"show_bounds adjustment failed: {e!r}")
            return out_ax

        _wrapped.__name__ = target.__name__
        _wrapped.__doc__ = (target.__doc__ or "") + "\n\nPatched: adds `show_bounds: bool` (default False)."
        setattr(sampling, target.__name__, _wrapped)
    except Exception as e:
        _safe_log(f"pypesto sampling bounds patch skipped: {e!r}")

def _patch_amici_species_guard():
    """
    amici 0.34.0: target amici.sbml_import.SbmlImporter.
    we auto-find a method whose source references 'Species' and wrap it so a
    variable only counts as a species if its string name is in symbols[SPECIES].
    """
    try:
        import inspect
        from amici.sbml_import import SbmlImporter, SymbolId  # <- your version exports these

        importer_cls = SbmlImporter

        # scan class methods to find one that likely classifies species
        candidates = []
        for name in dir(importer_cls):
            if name.startswith("__"):
                continue
            obj = getattr(importer_cls, name)
            if not callable(obj):
                continue
            try:
                src = inspect.getsource(obj)
            except Exception:
                continue
            # heuristic: classification logic references libsbml Species or the word 'Species'
            if "Species" in src or "sbml.Species" in src:
                candidates.append((name, obj, src))

        if not candidates:
            _safe_log("amici: no method mentioning 'Species' found on SbmlImporter; species guard not applied.")
            return

        # take the first reasonable candidate (works for 0.34.0); you can hard-pick later if needed
        target_name, original, _ = candidates[0]

        def _guard(self, *args, **kwargs):
            """
            wrapper: if original returns True for 'is species/state', enforce membership
            in self.symbols[SymbolId.SPECIES].
            """
            res = original(self, *args, **kwargs)
            try:
                # try to grab a 'symbol' argument (common pattern is (symbol, ...))
                symbol = args[0] if args else kwargs.get("symbol")
                symtab = getattr(self, "symbols", None)
                if symtab is None:
                    return res

                # figure out the key for species in the symbols table
                species_key = SymbolId.SPECIES if SymbolId in symtab else None
                if species_key is None:
                    for k in symtab.keys():
                        if str(k).upper().endswith("SPECIES") or str(k).upper() == "SPECIES":
                            species_key = k
                            break
                if species_key is None:
                    return res

                species_names = set(map(str, symtab[species_key]))

                # if original said True but the symbol string isn't listed as a species -> flip to False
                if isinstance(res, bool) and res:
                    if symbol is not None and str(symbol) not in species_names:
                        return False
            except Exception:
                # never crash the import on patch errors
                return res
            return res

        setattr(importer_cls, target_name, _guard)

    except Exception as e:
        _safe_log(f"amici species guard patch skipped: {e!r}")

def apply():
    _patch_petab_from_yaml()
    _patch_pypesto_importer_from_yaml()
    _patch_pypesto_sampling_bounds()
    _patch_amici_species_guard()