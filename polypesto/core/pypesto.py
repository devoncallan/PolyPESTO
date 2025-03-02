from typing import Dict, Tuple, List, Optional, Any, Union, TypeAlias
import os

from amici.petab.simulations import simulate_petab, rdatas_to_measurement_df
from amici.amici import Solver
from pypesto.petab import PetabImporter
from pypesto.problem import Problem
from pypesto import Result
import pypesto
import pypesto.optimize as optimize
import pypesto.petab
import pypesto.sample as sample
import pypesto.profile as profile
import pypesto.visualize as visualize

from polypesto.core.petab import PetabData, PetabIO
from polypesto.core.params import (
    ParameterGroup,
    ParameterSet,
    ParameterID,
    ParameterSetID,
)
from polypesto.models import sbml, ModelInterface
from experiments.irreversible_cpe.single_rxn.experiments import ExperimentData

# Type aliases for better code clarity
ExperimentName: TypeAlias = str
ExperimentCollection: TypeAlias = Dict[ExperimentName, ExperimentData]
ResultDict: TypeAlias = Dict[ParameterSetID, Result]
ExperimentResults: TypeAlias = Dict[ExperimentName, ResultDict]
StepConfigDict: TypeAlias = Dict[str, Any]
PEConfigDict: TypeAlias = Dict[str, StepConfigDict]


def set_amici_solver_params(problem: Problem) -> Problem:

    solver: Solver = problem.objective.amici_solver

    solver.setAbsoluteTolerance(1e-10)
    solver.setRelativeTolerance(1e-10)
    solver.setStabilityLimitFlag(True)
    # solver.setMaxSteps(1e5)
    # solver.setLinearSolver("dense")

    problem.objective.amici_solver = solver

    return problem


def load_pypesto_problem(
    yaml_path: str, model_name: str, **kwargs
) -> Tuple[PetabImporter, Problem]:

    importer = PetabImporter.from_yaml(
        yaml_path,
        model_name=model_name,
        base_path="", # Use absolute paths
    )
    problem = importer.create_problem(**kwargs)

    # problem = set_amici_solver_params(problem)
    # problem.objective.amici_solver.setPreequilibration(True)
    problem.objective.amici_solver.setNewtonMaxSteps(10000)
    problem.objective.amici_solver.setNewtonDampingFactorMode(2)
    problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem.objective.amici_solver.setRelativeTolerance(1e-6)
    # problem.objective.amici_solver.setStabilityLimitFlag(True)
    problem.objective.amici_solver.setMaxSteps(100000)
    problem.objective.amici_solver.setLinearSolver(6)
    # problem.objective.amici_solver.setStabilityLimitFlag(True)
    # problem.objective.amici_solver.setLinearMultistepMethod(1)

    # print(problem.objective.amici_solver.getNewtonMaxSteps())
    # print(problem.objective.amici_solver.getNewtonDampingFactorMode())
    # print(problem.objective.amici_solver.getAbsoluteTolerance())

    return importer, problem


def optimize_problem(
    problem: Problem, n_starts: int = 100, method: str = "Nelder-Mead", **kwargs
) -> Result:
    """Run optimization to find optimal parameter values.

    Parameters
    ----------
    problem : Problem
        Parameter estimation problem to solve
    n_starts : int, optional
        Number of optimization starts with different initial values, by default 100
    method : str, optional
        Optimization method to use, by default "Nelder-Mead"

    Returns
    -------
    Result
        Optimization result object containing best parameters and history
    """
    optimizer = optimize.ScipyOptimizer(method=method)
    history_options = pypesto.HistoryOptions(trace_record=True)
    engine = pypesto.engine.MultiProcessEngine()

    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
        engine=engine,
        history_options=history_options,
        **kwargs,
    )
    return result


def profile_problem(
    problem: Problem,
    method: str = "Nelder-Mead",
    result: Optional[Result] = None,
    **kwargs,
) -> Result:
    """Create profile likelihoods for parameters.

    Parameters
    ----------
    problem : Problem
        Parameter estimation problem
    method : str, optional
        Optimization method for profiling, by default "Nelder-Mead"
    result : Optional[Result], optional
        Previous optimization result to use as starting point, by default None

    Returns
    -------
    Result
        Updated result object containing parameter profiles
    """
    optimizer = optimize.ScipyOptimizer(method=method)
    result = profile.parameter_profile(
        problem=problem, result=result, optimizer=optimizer, **kwargs
    )
    return result


def sample_problem(
    problem: Problem,
    n_samples: int = 10000,
    n_chains: int = 3,
    result: Optional[Result] = None,
    **kwargs,
) -> Result:
    """Sample from the parameter posterior distribution.

    Parameters
    ----------
    problem : Problem
        Parameter estimation problem
    n_samples : int, optional
        Number of samples to generate, by default 10000
    n_chains : int, optional
        Number of parallel sampling chains, by default 3
    result : Optional[Result], optional
        Previous optimization result to use as starting point, by default None

    Returns
    -------
    Result
        Updated result object containing parameter samples
    """
    sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        n_chains=n_chains,
    )

    result = sample.sample(
        problem=problem, n_samples=n_samples, sampler=sampler, result=result, **kwargs
    )

    return result


def run_parameter_estimation(
    experiment_data: Union[ExperimentData, ExperimentCollection],
    config: Optional[PEConfigDict] = None,
) -> ExperimentResults:
    """Run parameter estimation with configurable settings.

    This function provides a flexible way to run parameter estimation steps
    (optimization, profiling, sampling) on one or multiple experiments.
    Only the steps specified in the config will be run.

    Parameters
    ----------
    experiment_data : Union[ExperimentData, ExperimentCollection]
        Either a single ExperimentData object or a dictionary of experiment name
        to ExperimentData objects (as returned by load_all_experiments)
    config : Optional[PEConfigDict], optional
        Dictionary containing configuration for different steps.
        Can contain keys 'optimize', 'profile', 'sample' with
        respective options for each function. If a key is absent,
        that step won't be run.
        Example: {'optimize': {'n_starts': 50, 'method': 'scipy-lbfgsb'}}

    Returns
    -------
    ExperimentResults
        Dictionary mapping experiment names to dictionaries of parameter set IDs to results.
        For single experiment inputs, the key will be "experiment".
    """
    config = config or {}
    results: ExperimentResults = {}

    # Handle single experiment or dictionary of experiments
    if isinstance(experiment_data, ExperimentData):
        experiments: ExperimentCollection = {"experiment": experiment_data}
    else:
        experiments = experiment_data

    # For each experiment
    for exp_name, exp_data in experiments.items():
        print(f"Running parameter estimation for experiment {exp_name}")

        results[exp_name] = {}
        petab_paths = exp_data.petab_paths

        # Initialize results dict in the experiment_data if needed
        if exp_data.results is None:
            exp_data.results = {}

        # Get all parameter sets
        for p_id, (importer, problem) in exp_data.problems.items():
            print(f"  Parameter set {p_id}")

            # Skip if no config steps are provided
            if not config:
                print(f"    No parameter estimation steps configured - skipping")
                continue

            # Initialize result to None and save components
            result: Optional[Result] = None
            save_components: Dict[str, bool] = {"problem": True}

            # Run optimization if configured
            if "optimize" in config:
                print(f"    Running optimization with {config['optimize']}")
                result = optimize_problem(problem, **config["optimize"])
                save_components["optimize"] = True

            # Skip to next parameter set if optimization not configured or failed
            if result is None:
                print(f"    No optimization result available - skipping further steps")
                continue

            # Run profiling if configured
            if "profile" in config:
                print(f"    Running profiling with {config['profile']}")
                result = profile_problem(problem, result=result, **config["profile"])
                save_components["profile"] = True

            # Run sampling if configured
            if "sample" in config:
                print(f"    Running sampling with {config['sample']}")
                result = sample_problem(problem, result=result, **config["sample"])
                save_components["sample"] = True

            # Store result and save to file
            results[exp_name][p_id] = result

            print(f"    Saving results to {petab_paths.pypesto_results(p_id)}")
            pypesto.store.write_result(
                result=result,
                filename=petab_paths.pypesto_results(p_id),
                overwrite=True,
                **save_components,
            )

            # Always update the experiment_data object's results
            exp_data.results[p_id] = result

    return results

#########################
### Write PETab files ###
#########################


# def write_initial_petab(
#     model_def: sbml.ModelDefinition,
#     pg: ParameterGroup,
#     data: PetabData,
#     dir: str,
# ) -> PetabPaths:
#     """Write initial PEtab files without simulated data.

#     Parameters
#     ----------
#     model_def : sbml.ModelDefinition
#         SBML model definition
#     pg : ParameterGroup
#         Parameter group with true parameters
#     data : PetabData
#         Petab data (observables, conditions, measurements, parameters)
#     dir : str
#         Directory to write files to

#     Returns
#     -------
#     PetabPaths
#         Object containing paths to all created files
#     """
#     paths = PetabPaths(dir)

#     sbml_filepath = sbml.write_model(model_def=model_def, model_dir=paths.common_dir)

#     PetabIO.write_obs_df(data.obs_df, filename=paths.observables)
#     PetabIO.write_cond_df(data.cond_df, filename=paths.conditions)
#     PetabIO.write_param_df(data.param_df, filename=paths.fit_parameters)
#     pg.write(paths.true_params)

#     for p_id in pg.get_ids():
#         paths.make_exp_dir(p_id)

#         # Write the true parameters to file
#         pg.by_id(p_id).write(paths.params(p_id))

#         PetabIO.write_meas_df(data.meas_df, filename=paths.measurements(p_id))
#         PetabIO.write_yaml(
#             yaml_filepath=str(paths.petab_yaml(p_id)),
#             sbml_filepath=sbml_filepath,
#             cond_filepath=paths.conditions,
#             meas_filepath=paths.measurements(p_id),
#             obs_filepath=paths.observables,
#             param_filepath=paths.fit_parameters,
#         )

#     return paths


# def create_problem_set(
#     model: ModelInterface,
#     pg: ParameterGroup,
#     data: PetabData,
#     dir: str,
#     force_compile: bool = False,
# ) -> PetabPaths:
#     """Create Petab problem set by simulating data.

#     Parameters
#     ----------
#     model : ModelInterface
#         Model interface to use for simulation
#     pg : ParameterGroup
#         Parameter group (multiple sets of parameters) to generate data
#     data : PetabData
#         Contains observables, conditions, measurements, fit params
#     dir : str
#         Directory to write files to
#     force_compile : bool, optional
#         Force recompilation of model, by default False

#     Returns
#     -------
#     PetabPaths
#         Object containing paths to all created files
#     """

#     model_name = model.name
#     os.makedirs(os.path.dirname(dir), exist_ok=True)
#     os.makedirs(dir, exist_ok=True)

#     # Write without simulated data first
#     paths = write_initial_petab(model.sbml_model_def(), pg, data, dir=dir)

#     yaml_paths = paths.find_yaml_paths()
#     yaml_path = list(yaml_paths.values())[0]

#     importer, problem = load_pypesto_problem(
#         yaml_path, str(model_name), force_compile=force_compile
#     )

#     for p_id, yaml_path in yaml_paths.items():
#         print(f"Simulating data for {p_id}...")

#         params_path = paths.params(p_id)
#         params = ParameterSet.load(params_path).to_dict()

#         sim_data = simulate_petab(
#             petab_problem=importer.petab_problem,
#             amici_model=problem.objective.amici_model,
#             solver=problem.objective.amici_solver,
#             problem_parameters=params,
#         )
#         meas_df = rdatas_to_measurement_df(
#             sim_data["rdatas"],
#             problem.objective.amici_model,
#             importer.petab_problem.measurement_df,
#         )

#         PetabIO.write_meas_df(meas_df, filename=paths.measurements(p_id))

#     return paths
