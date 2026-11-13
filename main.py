import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mealpy import BinaryVar, Multitask

from imports import *
from helpers import *
from MyProblem import MyProblem

from arbos import ARBOS
from binary_de import BinaryDE
from binary_ga import BinaryGA
from binary_gwo import BinaryGWO
from binary_pso import BinaryPSO


# ============================================================
# Experiment Configuration
# ============================================================

RUN_FULL_EXPERIMENT = True

N_TRIALS = 5
SMOKE_TEST_TRIALS = 2

N_JOBS = 3

RANDOM_SEED = RANDOM_STATE

HISTORY_FOLDER = Path("history")
EXPERIMENTS_FOLDER = HISTORY_FOLDER / "experiments"

RUN_ID = os.environ.get(
    "FS_MHA_RUN_ID",
    "unassigned",
)

RUN_OUTPUT_FOLDER = EXPERIMENTS_FOLDER / RUN_ID

BEST_FIT_FOLDER = RUN_OUTPUT_FOLDER / "best_fit"
CONVERGENCE_FOLDER = RUN_OUTPUT_FOLDER / "convergence"
RUN_DETAILS_FOLDER = RUN_OUTPUT_FOLDER / "run_details"
OUTPUT_FOLDER = RUN_OUTPUT_FOLDER / "plots"

RESULT_FILENAME = RUN_OUTPUT_FOLDER / "results.xlsx"

DATASET_NAMES = [
    name.lower()
    for name in DATASET_NAMES
]

if RUN_FULL_EXPERIMENT:

    SELECTED_DATASETS = DATASET_NAMES
    NUMBER_OF_TRIALS = N_TRIALS

else:

    SELECTED_DATASETS = DATASET_NAMES[:1]
    NUMBER_OF_TRIALS = SMOKE_TEST_TRIALS


# ============================================================
# Utility Functions
# ============================================================

def json_serializer(value):

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def prepare_run_folders():

    folders = [
        RUN_OUTPUT_FOLDER,
        BEST_FIT_FOLDER,
        CONVERGENCE_FOLDER,
        RUN_DETAILS_FOLDER,
        OUTPUT_FOLDER,
    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_run_details(record):

    algorithm_name = record["algorithm"]
    dataset_name = record["dataset"]
    trial_number = record["trial"]

    output_folder = (
        RUN_DETAILS_FOLDER
        / algorithm_name
        / dataset_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_folder
        / f"trial_{trial_number:03d}.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            record,
            file,
            indent=4,
            ensure_ascii=False,
            default=json_serializer,
        )


# ============================================================
# Algorithm Factory
# ============================================================

def build_algorithms():

    algorithm_factories = {
        "ARBOS": lambda: ARBOS(
            epoch=EPOCH,
            pop_size=POP_SIZE,
        ),

        "BinaryDE": lambda: BinaryDE(
            epoch=EPOCH,
            pop_size=POP_SIZE,
        ),

        "BinaryGA": lambda: BinaryGA(
            epoch=EPOCH,
            pop_size=POP_SIZE,
        ),

        "BinaryGWO": lambda: BinaryGWO(
            epoch=EPOCH,
            pop_size=POP_SIZE,
        ),

        "BinaryPSO": lambda: BinaryPSO(
            epoch=EPOCH,
            pop_size=POP_SIZE,
        ),
    }

    selected_algorithms = []

    for algorithm_name in ALGORITHMS:

        if algorithm_name not in algorithm_factories:
            raise ValueError(
                f"Unknown algorithm in Config.py: {algorithm_name}"
            )

        algorithm = algorithm_factories[algorithm_name]()
        selected_algorithms.append(algorithm)

    return selected_algorithms


# ============================================================
# Problem Factory
# ============================================================

def build_problems(dataset_names):

    problems = []
    dataset_information = {}

    for dataset_name in dataset_names:

        print("\n" + "=" * 70)
        print(f"Preparing dataset: {dataset_name}")
        print("=" * 70)

        data = convert_datasets(
            dataset_name,
            random_state=RANDOM_SEED,
        )

        n_features = data["X_train"].shape[1]

        bounds = BinaryVar(
            n_vars=n_features,
        )

        problem = MyProblem(
            bounds=bounds,
            minmax=MIN_MAX_PROBLEM,
            name=dataset_name,
            data=data,
            log_to=None,
            classifier=CLASSIFIER,
            average_metric=AVERAGE_METRIC,
            random_state=RANDOM_SEED,
            alpha=0.8,
            feature_weight=0.2,
            knn_neighbors=5,
        )

        problem.feature_names = data.get(
            "feature_names",
            [],
        )

        problems.append(problem)

        dataset_information[dataset_name] = {
            "n_features": n_features,
            "n_train": len(data["y_train"]),
            "n_validation": len(data["y_validation"]),
            "n_test": len(data["y_test"]),
            "feature_names": data.get(
                "feature_names",
                [],
            ),
            "split_info": data["split_info"],
        }

        print(f"Features: {n_features}")

        print(
            f"Training samples: "
            f"{len(data['y_train'])}"
        )

        print(
            f"Validation samples: "
            f"{len(data['y_validation'])}"
        )

        print(
            f"Test samples: "
            f"{len(data['y_test'])}"
        )

    return problems, dataset_information


# ============================================================
# Mealpy Multitask Configuration
# ============================================================

def create_multitask_modes(
    n_algorithms,
    n_problems,
):

    # Compatible with the installed Mealpy version.
    return ["single"]


# ============================================================
# Collect Detailed Results
# ============================================================

def collect_run_details():

    records = []

    for file_path in RUN_DETAILS_FOLDER.rglob(
        "trial_*.json"
    ):

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            records.append(
                json.load(file)
            )

    if not records:
        raise RuntimeError(
            "No detailed run results were found."
        )

    results_df = pd.DataFrame(records)

    results_df.sort_values(
        by=[
            "dataset",
            "algorithm",
            "trial",
        ],
        inplace=True,
    )

    output_file = (
        RUN_DETAILS_FOLDER / "run_details.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    return results_df


# ============================================================
# Validate Detailed Trial Results
# ============================================================

def validate_run_details(
    results_df,
    datasets,
    algorithms,
    expected_trials,
):

    expected_records = (
        len(datasets)
        * len(algorithms)
        * expected_trials
    )

    if len(results_df) != expected_records:

        raise RuntimeError(
            "Unexpected number of trial records.\n"
            f"Expected: {expected_records}\n"
            f"Found: {len(results_df)}"
        )

    duplicated = results_df.duplicated(
        subset=[
            "dataset",
            "algorithm",
            "trial",
        ]
    )

    if duplicated.any():

        raise RuntimeError(
            "Duplicate trial records were detected."
        )

    expected_pairs = {
        (dataset, algorithm)
        for dataset in datasets
        for algorithm in algorithms
    }

    actual_pairs = set(
        zip(
            results_df["dataset"],
            results_df["algorithm"],
        )
    )

    missing_pairs = expected_pairs - actual_pairs

    if missing_pairs:

        raise RuntimeError(
            f"Missing algorithm/dataset results: "
            f"{sorted(missing_pairs)}"
        )

    for (dataset, algorithm), group in results_df.groupby(
        ["dataset", "algorithm"]
    ):

        if len(group) != expected_trials:

            raise RuntimeError(
                f"{dataset}/{algorithm}: expected "
                f"{expected_trials} trials, "
                f"found {len(group)}."
            )

        actual_trials = set(
            group["trial"].astype(int)
        )

        expected_trial_ids = set(
            range(1, expected_trials + 1)
        )

        if actual_trials != expected_trial_ids:

            raise RuntimeError(
                f"Incorrect trial IDs for "
                f"{dataset}/{algorithm}: "
                f"{sorted(actual_trials)}"
            )

    print(
        "\nTrial-level result validation passed."
    )


# ============================================================
# Extended Mealpy Multitask
# ============================================================

class FeatureSelectionMultitask(Multitask):

    """
    Extends Mealpy Multitask.

    Additional outputs:
        - Best binary mask
        - Selected feature indices
        - Selected feature names
        - Final Test metrics
        - Optimization execution time
        - Test evaluation time
    """

    def __run__(
        self,
        id_trial,
        optimizer,
        problem,
        termination=None,
        mode="single",
    ):

        # ----------------------------------------------------
        # 1. Optimization
        # ----------------------------------------------------

        optimization_start = time.perf_counter()

        g_best = optimizer.solve(
            problem,
            mode=mode,
            termination=termination,
            n_workers=self.n_workers,
        )

        optimization_time = (
            time.perf_counter()
            - optimization_start
        )

        # ----------------------------------------------------
        # 2. Extract best binary solution
        # ----------------------------------------------------

        best_mask = np.asarray(
            g_best.solution
        ).reshape(-1)

        if not np.all(
            np.isin(
                best_mask,
                [0, 1, False, True],
            )
        ):

            raise ValueError(
                "The optimizer returned "
                "a non-binary solution."
            )

        best_mask = best_mask.astype(
            np.int8
        )

        selected_indices = np.flatnonzero(
            best_mask
        ).tolist()

        feature_names = getattr(
            problem,
            "feature_names",
            [],
        )

        if len(feature_names) == len(best_mask):

            selected_feature_names = [
                str(feature_names[index])
                for index in selected_indices
            ]

        else:

            selected_feature_names = [
                f"feature_{index}"
                for index in selected_indices
            ]

        # ----------------------------------------------------
        # 3. Final Test evaluation
        # ----------------------------------------------------

        test_metrics = None
        test_evaluation_error = None
        test_evaluation_time = None

        if len(selected_indices) == 0:

            test_evaluation_error = (
                "The optimizer returned "
                "an empty feature subset."
            )

        else:

            test_start = time.perf_counter()

            try:

                test_metrics = problem.evaluate_test(
                    best_mask
                )

            except Exception as error:

                test_evaluation_error = (
                    f"{type(error).__name__}: {error}"
                )

            test_evaluation_time = (
                time.perf_counter()
                - test_start
            )

        # ----------------------------------------------------
        # 4. Create trial record
        # ----------------------------------------------------

        record = {

            "dataset": problem.get_name(),

            "algorithm": (
                optimizer.__class__.__name__
            ),

            "trial": int(id_trial),

            "best_fitness": float(
                g_best.target.fitness
            ),

            "best_mask": best_mask.tolist(),

            "selected_feature_indices": (
                selected_indices
            ),

            "selected_feature_names": (
                selected_feature_names
            ),

            "n_selected_features": (
                len(selected_indices)
            ),

            "n_total_features": len(best_mask),

            "optimization_time": (
                optimization_time
            ),

            "test_evaluation_time": (
                test_evaluation_time
            ),

            "test_metrics": test_metrics,

            "test_evaluation_error": (
                test_evaluation_error
            ),
        }

        save_run_details(record)

        # ----------------------------------------------------
        # 5. Return Mealpy-compatible results
        # ----------------------------------------------------

        return {

            "id_trial": id_trial,

            "best_fitness": (
                g_best.target.fitness
            ),

            "convergence": (
                optimizer.history.list_global_best_fit
            ),

            "problem_name": (
                optimizer.problem.get_name()
            ),
        }


# ============================================================
# Experiment Audit Table
# ============================================================

def create_experiment_audit(
    stats_df,
    test_performance_df,
    expected_datasets,
    algorithms,
    expected_trials,
):

    rows = []

    for dataset in expected_datasets:

        for algorithm in algorithms:

            fitness_row = stats_df[
                (stats_df["Function"] == dataset)
                & (stats_df["Algorithm"] == algorithm)
            ]

            test_row = test_performance_df[
                (test_performance_df["Dataset"] == dataset)
                & (test_performance_df["Algorithm"] == algorithm)
            ]

            n_fitness_trials = (
                int(fitness_row.iloc[0]["N Trials"])
                if not fitness_row.empty
                else 0
            )

            n_test_trials = (
                int(test_row.iloc[0]["N Trials"])
                if not test_row.empty
                else 0
            )

            rows.append({

                "Dataset": dataset,
                "Algorithm": algorithm,

                "Expected Trials": expected_trials,

                "Fitness Trials": n_fitness_trials,

                "Test Trials": n_test_trials,

                "Fitness Complete": (
                    n_fitness_trials == expected_trials
                ),

                "Test Complete": (
                    n_test_trials == expected_trials
                ),

            })

    audit_df = pd.DataFrame(rows)

    if not audit_df[
        ["Fitness Complete", "Test Complete"]
    ].all().all():

        raise RuntimeError(
            "Experiment audit detected incomplete results."
        )

    return audit_df


# ============================================================
# Main Execution
# ============================================================

def main():

    global RUN_ID
    global RUN_OUTPUT_FOLDER
    global BEST_FIT_FOLDER
    global CONVERGENCE_FOLDER
    global RUN_DETAILS_FOLDER
    global OUTPUT_FOLDER
    global RESULT_FILENAME

    # --------------------------------------------------------
    # 1. Generate unique experiment ID
    # --------------------------------------------------------

    RUN_ID = (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_{time.time_ns() % 1_000_000_000:09d}"
    )

    os.environ["FS_MHA_RUN_ID"] = RUN_ID

    # --------------------------------------------------------
    # 2. Configure output paths
    # --------------------------------------------------------

    RUN_OUTPUT_FOLDER = (
        EXPERIMENTS_FOLDER / RUN_ID
    )

    BEST_FIT_FOLDER = (
        RUN_OUTPUT_FOLDER / "best_fit"
    )

    CONVERGENCE_FOLDER = (
        RUN_OUTPUT_FOLDER / "convergence"
    )

    RUN_DETAILS_FOLDER = (
        RUN_OUTPUT_FOLDER / "run_details"
    )

    OUTPUT_FOLDER = (
        RUN_OUTPUT_FOLDER / "plots"
    )

    RESULT_FILENAME = (
        RUN_OUTPUT_FOLDER / "results.xlsx"
    )

    # --------------------------------------------------------
    # 3. Experiment information
    # --------------------------------------------------------

    print("\n" + "#" * 70)
    print("FEATURE SELECTION EXPERIMENT")
    print("#" * 70)

    print(f"Experiment ID: {RUN_ID}")
    print(f"Datasets: {len(SELECTED_DATASETS)}")
    print(f"Algorithms: {len(ALGORITHMS)}")

    print(
        f"Trials per algorithm/dataset: "
        f"{NUMBER_OF_TRIALS}"
    )

    print(f"Epoch: {EPOCH}")
    print(f"Population size: {POP_SIZE}")

    print(
        f"Experiment output: {RUN_OUTPUT_FOLDER}"
    )

    # --------------------------------------------------------
    # 4. Create folders
    # --------------------------------------------------------

    HISTORY_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepare_run_folders()

    # --------------------------------------------------------
    # 5. Build optimization problems
    # --------------------------------------------------------

    problems, dataset_information = build_problems(
        SELECTED_DATASETS
    )

    # --------------------------------------------------------
    # 6. Build algorithms
    # --------------------------------------------------------

    algorithms = build_algorithms()

    algorithm_names = [
        algorithm.__class__.__name__
        for algorithm in algorithms
    ]

    if algorithm_names != ALGORITHMS:

        raise RuntimeError(
            "Algorithm configuration mismatch.\n"
            f"Configured: {ALGORITHMS}\n"
            f"Instantiated: {algorithm_names}"
        )

    if PROPOSED_ALGORITHM not in ALGORITHMS:

        raise ValueError(
            f"Proposed algorithm '{PROPOSED_ALGORITHM}' "
            "is not included in ALGORITHMS."
        )

    # --------------------------------------------------------
    # 7. Configure Mealpy Multitask
    # --------------------------------------------------------

    modes = create_multitask_modes(
        n_algorithms=len(algorithms),
        n_problems=len(problems),
    )

    available_cpus = os.cpu_count() or 2

    maximum_workers = min(
        61,
        available_cpus - 1,
    )

    if maximum_workers < 2:

        raise RuntimeError(
            "Mealpy multiprocessing requires "
            "at least two workers."
        )

    n_jobs = min(
        N_JOBS,
        maximum_workers,
    )

    print(f"Available CPUs: {available_cpus}")
    print(f"Parallel workers: {n_jobs}")

    multitask = FeatureSelectionMultitask(
        algorithms=algorithms,
        problems=problems,
        modes=modes,
        n_workers=None,
    )

    # --------------------------------------------------------
    # 8. Execute independent trials
    # --------------------------------------------------------

    print("\nStarting Mealpy Multitask execution...")

    start_time = time.perf_counter()

    multitask.execute(
        n_trials=NUMBER_OF_TRIALS,
        n_jobs=n_jobs,
        save_path=str(RUN_OUTPUT_FOLDER),
        save_as="csv",
        save_convergence=True,
        verbose=True,
    )

    total_execution_time = (
        time.perf_counter() - start_time
    )

    print("\nMultitask execution completed.")

    print(
        f"Total execution time: "
        f"{total_execution_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # 9. Collect detailed trial results
    # --------------------------------------------------------

    results_df = collect_run_details()

    validate_run_details(
        results_df,
        SELECTED_DATASETS,
        ALGORITHMS,
        NUMBER_OF_TRIALS,
    )

    print("\nDetailed trial results:")

    print(
        results_df[
            [
                "dataset",
                "algorithm",
                "trial",
                "best_fitness",
                "n_selected_features",
                "optimization_time",
            ]
        ].to_string(index=False)
    )

    print(
        "\nDetailed results saved to: "
        f"{RUN_DETAILS_FOLDER / 'run_details.csv'}"
    )

    # --------------------------------------------------------
    # 10. Statistical analysis
    # --------------------------------------------------------

    if RUN_FULL_EXPERIMENT:

        print("\n" + "#" * 70)
        print("STATISTICAL ANALYSIS")
        print("#" * 70)

        # ----------------------------------------------------
        # 10.1 Read fitness results
        # ----------------------------------------------------

        stats_df = read_algorithm_data(
            BEST_FIT_FOLDER,
            ALGORITHMS,
            expected_trials=NUMBER_OF_TRIALS,
            expected_datasets=SELECTED_DATASETS,
        )

        # ----------------------------------------------------
        # 10.2 Validate complete fitness matrix
        # ----------------------------------------------------

        fitness_matrix = validate_experiment_completeness(
            stats_df,
            SELECTED_DATASETS,
            ALGORITHMS,
            NUMBER_OF_TRIALS,
        )

        print("\nFitness result matrix:")
        print(fitness_matrix)

        # ----------------------------------------------------
        # 10.3 Fitness summary
        # ----------------------------------------------------

        performance_df = prepare_performance_data(
            stats_df
        )

        # ----------------------------------------------------
        # 10.4 Test performance summary
        # ----------------------------------------------------

        test_performance_df = prepare_test_performance_data(
            results_df,
            expected_datasets=SELECTED_DATASETS,
            algorithms=ALGORITHMS,
            expected_trials=NUMBER_OF_TRIALS,
        )

        # ----------------------------------------------------
        # 10.5 Runtime analysis
        # ----------------------------------------------------

        runtime_df = prepare_runtime_data(
            results_df,
            expected_datasets=SELECTED_DATASETS,
            algorithms=ALGORITHMS,
            expected_trials=NUMBER_OF_TRIALS,
        )

        print("\nRuntime Analysis:")
        print(
            runtime_df[
                [
                    "Dataset",
                    "Algorithm",
                    "Optimization Mean (s)",
                    "Optimization STD (s)",
                    "Test Evaluation Mean (s)",
                    "Total Mean (s)",
                ]
            ].to_string(index=False)
        )

        # ----------------------------------------------------
        # 10.6 Experiment audit
        # ----------------------------------------------------

        completeness_df = create_experiment_audit(
            stats_df,
            test_performance_df,
            SELECTED_DATASETS,
            ALGORITHMS,
            NUMBER_OF_TRIALS,
        )

        # ----------------------------------------------------
        # 10.7 Friedman test
        # ----------------------------------------------------

        friedman_results = perform_friedman_test(
            stats_df,
            expected_datasets=SELECTED_DATASETS,
            algorithms=ALGORITHMS,
        )

        print("\nFriedman test:")
        print(
            f"Statistic: {friedman_results[0]:.6f}"
        )
        print(
            f"p-value: {friedman_results[1]:.8e}"
        )

        print("\nFriedman mean ranks:")

        for algorithm, rank in sorted(
            friedman_results[2].items(),
            key=lambda item: item[1],
        ):

            print(
                f"{algorithm:<15} {rank:.4f}"
            )

        # ----------------------------------------------------
        # 10.8 Wilcoxon with Holm correction
        # ----------------------------------------------------

        wilcoxon_results = perform_wilcoxon_test(
            stats_df,
            PROPOSED_ALGORITHM,
        )

        print("\nWilcoxon-Holm results:")

        for algorithm, result in (
            wilcoxon_results["Overall"].items()
        ):

            print(
                f"{PROPOSED_ALGORITHM} vs {algorithm}: "
                f"raw p={result['Raw p-value']:.6g}, "
                f"Holm p={result['Holm Adjusted p-value']:.6g}, "
                f"result={result['Better Algorithm']}"
            )

        # ----------------------------------------------------
        # 10.9 Nemenyi post-hoc test
        # ----------------------------------------------------

        nemenyi_results = perform_nemenyi_test(
            friedman_results,
            n_datasets=len(SELECTED_DATASETS),
        )

        print("\nNemenyi Critical Difference:")
        print(
            f"CD = {nemenyi_results['Critical Difference']:.6f}"
        )

        print("\nNemenyi comparisons:")

        print(
            nemenyi_results["Comparisons"].to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # 10.10 Excel report
        # ----------------------------------------------------

        save_to_excel(
            performance_df=performance_df,
            test_performance_df=test_performance_df,
            runtime_df=runtime_df,
            friedman_results=friedman_results,
            wilcoxon_results=wilcoxon_results,
            nemenyi_results=nemenyi_results,
            completeness_df=completeness_df,
            filename=RESULT_FILENAME,
        )

        print(
            f"\nExcel report saved to: {RESULT_FILENAME}"
        )

        # ----------------------------------------------------
        # 10.11 Convergence analysis
        # ----------------------------------------------------

        convergence_data = read_convergence_data(
            CONVERGENCE_FOLDER,
            SELECTED_DATASETS,
            ALGORITHMS,
        )

        plot_convergence_curves(
            convergence_data,
            SELECTED_DATASETS,
            ALGORITHMS,
            OUTPUT_FOLDER,
        )

        # ----------------------------------------------------
        # 10.12 Friedman rank plot
        # ----------------------------------------------------

        plot_friedman_ranks(
            friedman_results[2],
            OUTPUT_FOLDER,
        )

        # ----------------------------------------------------
        # 10.13 Critical Difference diagram
        # ----------------------------------------------------

        plot_critical_difference(
            friedman_results,
            nemenyi_results,
            OUTPUT_FOLDER,
        )

        # ----------------------------------------------------
        # 10.14 Wilcoxon heatmaps
        # ----------------------------------------------------

        plot_wilcoxon_heatmaps(
            wilcoxon_results,
            PROPOSED_ALGORITHM,
            OUTPUT_FOLDER,
        )

        # ----------------------------------------------------
        # 10.15 Win / Tie / Loss
        # ----------------------------------------------------

        plot_pairwise_win_tie_loss(
            wilcoxon_results,
            proposed_algo=PROPOSED_ALGORITHM,
            save_path=OUTPUT_FOLDER,
        )

        # ----------------------------------------------------
        # 10.16 Runtime analysis plots
        # ----------------------------------------------------

        plot_runtime_analysis(
            runtime_df=runtime_df,
            run_details_df=results_df,
            datasets=SELECTED_DATASETS,
            algorithms=ALGORITHMS,
            output_folder=OUTPUT_FOLDER,
        )

        # ----------------------------------------------------
        # 10.17 Final report
        # ----------------------------------------------------

        print("\n" + "#" * 70)
        print("EXPERIMENT FINISHED SUCCESSFULLY")
        print("#" * 70)

        print(
            f"Datasets: {len(SELECTED_DATASETS)}"
        )

        print(
            f"Algorithms: {len(ALGORITHMS)}"
        )

        print(
            f"Trials per combination: {NUMBER_OF_TRIALS}"
        )

        print(
            f"Total independent trials: "
            f"{len(SELECTED_DATASETS) * len(ALGORITHMS) * NUMBER_OF_TRIALS}"
        )

        print(
            f"Excel report: {RESULT_FILENAME}"
        )

        print(
            f"Plots folder: {OUTPUT_FOLDER}"
        )

    # --------------------------------------------------------
    # 11. Smoke-test mode
    # --------------------------------------------------------

    else:

        print("\nStatistical analysis skipped.")

        print(
            "Smoke-test results are not suitable "
            "for statistical comparison."
        )

        print(
            "Detailed trial records were validated "
            "successfully."
        )

    # --------------------------------------------------------
    # 12. Final output
    # --------------------------------------------------------

    print("\nExperiment output directory:")
    print(RUN_OUTPUT_FOLDER)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()