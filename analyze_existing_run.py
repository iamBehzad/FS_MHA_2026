
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Config import (
    ALGORITHMS,
    PROPOSED_ALGORITHM,
)

from helpers import (
    read_algorithm_data,
    prepare_performance_data,
    prepare_test_performance_data,
    prepare_runtime_data,
    perform_friedman_test,
    perform_wilcoxon_test,
    perform_nemenyi_test,
    save_to_excel,
    read_convergence_data,
    plot_convergence_curves,
    plot_friedman_ranks,
    plot_critical_difference,
    plot_wilcoxon_heatmaps,
    plot_pairwise_win_tie_loss,
    plot_runtime_analysis,
)


# ============================================================
# Experiment configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

EXPERIMENTS_FOLDER = (
    PROJECT_ROOT / "history" / "experiments"
)

# Pilot experiment configuration.
# These values describe the existing experiment,
# not the configuration of a future experiment.

EXPECTED_TRIALS = 5

SELECTED_DATASETS = [
    "breast",
    "ionosphere",
    "sonar",
]

SELECTED_ALGORITHMS = list(ALGORITHMS)

# Existing pilot experiment ID.
DEFAULT_RUN_ID = "20261113_192819_314166000"

# Output filenames
EXCEL_FILENAME = "results.xlsx"

PLOT_FOLDER_NAME = "plots"

ALPHA = 0.05


# ============================================================
# Experiment paths
# ============================================================

def get_experiment_paths(run_id):

    run_folder = EXPERIMENTS_FOLDER / run_id

    if not run_folder.exists():
        raise FileNotFoundError(
            f"Experiment folder does not exist:\n{run_folder}"
        )

    paths = {
        "root": run_folder,
        "best_fit": run_folder / "best_fit",
        "convergence": run_folder / "convergence",
        "run_details": run_folder / "run_details",
        "plots": run_folder / PLOT_FOLDER_NAME,
        "excel": run_folder / EXCEL_FILENAME,
    }

    required_folders = [
        "best_fit",
        "convergence",
        "run_details",
    ]

    for folder_name in required_folders:

        folder = paths[folder_name]

        if not folder.exists():
            raise FileNotFoundError(
                f"Required folder not found: {folder}"
            )

    paths["plots"].mkdir(
        parents=True,
        exist_ok=True,
    )

    return paths


# ============================================================
# Read existing trial records
# ============================================================

def collect_existing_run_details(run_details_folder):

    records = []

    json_files = sorted(
        Path(run_details_folder).rglob("trial_*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            "No trial JSON files were found in:\n"
            f"{run_details_folder}"
        )

    for file_path in json_files:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as file:

                record = json.load(file)

            required_fields = {
                "dataset",
                "algorithm",
                "trial",
                "best_fitness",
                "optimization_time",
                "test_evaluation_time",
                "test_metrics",
                "test_evaluation_error",
            }

            missing = required_fields - set(record.keys())

            if missing:
                raise ValueError(
                    f"Missing fields: {sorted(missing)}"
                )

            records.append(record)

        except Exception as exc:

            raise RuntimeError(
                f"Could not read trial file:\n"
                f"{file_path}\n"
                f"Reason: {exc}"
            ) from exc

    results_df = pd.DataFrame(records)

    print(
        f"\nTrial JSON files loaded: {len(results_df)}"
    )

    return results_df


# ============================================================
# Validate trial-level records
# ============================================================

def validate_existing_run_details(
    results_df,
    datasets,
    algorithms,
    expected_trials,
):

    required_columns = {
        "dataset",
        "algorithm",
        "trial",
        "best_fitness",
        "optimization_time",
        "test_evaluation_time",
        "test_metrics",
        "test_evaluation_error",
    }

    missing_columns = (
        required_columns - set(results_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in trial records: "
            f"{sorted(missing_columns)}"
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
    unexpected_pairs = actual_pairs - expected_pairs

    if missing_pairs or unexpected_pairs:

        raise RuntimeError(
            "Experiment coverage validation failed.\n"
            f"Missing pairs: {sorted(missing_pairs)}\n"
            f"Unexpected pairs: {sorted(unexpected_pairs)}"
        )

    if results_df.duplicated(
        subset=["dataset", "algorithm", "trial"]
    ).any():

        raise RuntimeError(
            "Duplicate dataset/algorithm/trial records found."
        )

    # Check trial numbering and count.
    for dataset in datasets:

        for algorithm in algorithms:

            group = results_df[
                (results_df["dataset"] == dataset)
                & (results_df["algorithm"] == algorithm)
            ]

            if len(group) != expected_trials:

                raise RuntimeError(
                    f"{dataset}/{algorithm}: expected "
                    f"{expected_trials} trials, "
                    f"found {len(group)}."
                )

            actual_trials = sorted(
                pd.to_numeric(
                    group["trial"],
                    errors="raise",
                ).astype(int).tolist()
            )

            expected_trial_ids = list(
                range(1, expected_trials + 1)
            )

            if actual_trials != expected_trial_ids:

                raise RuntimeError(
                    f"Invalid trial identifiers for "
                    f"{dataset}/{algorithm}.\n"
                    f"Expected: {expected_trial_ids}\n"
                    f"Found: {actual_trials}"
                )

    # Validate best fitness.
    fitness_values = pd.to_numeric(
        results_df["best_fitness"],
        errors="coerce",
    )

    if (
        fitness_values.isna().any()
        or not np.isfinite(fitness_values).all()
    ):

        raise ValueError(
            "Invalid or non-finite best_fitness values."
        )

    # Check Test evaluation errors.
    errors = results_df[
        "test_evaluation_error"
    ].dropna()

    if not errors.empty:

        raise RuntimeError(
            "Test evaluation errors were found:\n"
            + errors.to_string(index=False)
        )

    print("\nTrial-level validation passed.")

    print(
        f"Datasets: {len(datasets)}"
    )

    print(
        f"Algorithms: {len(algorithms)}"
    )

    print(
        f"Trials per combination: {expected_trials}"
    )

    print(
        f"Total independent trials: "
        f"{len(datasets) * len(algorithms) * expected_trials}"
    )


# ============================================================
# Create experiment audit
# ============================================================

def create_experiment_audit(
    stats_df,
    test_performance_df,
    runtime_df,
    datasets,
    algorithms,
    expected_trials,
):

    rows = []

    for dataset in datasets:

        for algorithm in algorithms:

            fitness_row = stats_df[
                (stats_df["Function"] == dataset)
                & (stats_df["Algorithm"] == algorithm)
            ]

            test_row = test_performance_df[
                (test_performance_df["Dataset"] == dataset)
                & (test_performance_df["Algorithm"] == algorithm)
            ]

            runtime_row = runtime_df[
                (runtime_df["Dataset"] == dataset)
                & (runtime_df["Algorithm"] == algorithm)
            ]

            fitness_trials = (
                int(fitness_row.iloc[0]["N Trials"])
                if len(fitness_row) == 1
                else 0
            )

            test_trials = (
                int(test_row.iloc[0]["N Trials"])
                if len(test_row) == 1
                else 0
            )

            runtime_trials = (
                int(runtime_row.iloc[0]["N Trials"])
                if len(runtime_row) == 1
                else 0
            )

            complete = (
                fitness_trials == expected_trials
                and test_trials == expected_trials
                and runtime_trials == expected_trials
            )

            rows.append({
                "Dataset": dataset,
                "Algorithm": algorithm,
                "Expected Trials": expected_trials,
                "Fitness Trials": fitness_trials,
                "Test Trials": test_trials,
                "Runtime Trials": runtime_trials,
                "Complete": complete,
            })

    audit_df = pd.DataFrame(rows)

    if not audit_df["Complete"].all():

        raise RuntimeError(
            "Experiment audit failed:\n"
            + audit_df.to_string(index=False)
        )

    return audit_df


# ============================================================
# Print runtime summary
# ============================================================

def print_runtime_summary(runtime_df):

    columns = [
        "Dataset",
        "Algorithm",
        "Optimization Mean (s)",
        "Optimization STD (s)",
        "Test Evaluation Mean (s)",
        "Total Mean (s)",
    ]

    print("\nRuntime Analysis Summary:")

    print(
        runtime_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


# ============================================================
# Main analysis workflow
# ============================================================

def analyze_existing_run(run_id):

    print("\n" + "=" * 75)
    print("ANALYSIS OF EXISTING FEATURE SELECTION EXPERIMENT")
    print("=" * 75)

    print(f"Experiment ID: {run_id}")

    print(f"Project root: {PROJECT_ROOT}")

    print(f"Expected trials: {EXPECTED_TRIALS}")

    print(f"Datasets: {SELECTED_DATASETS}")

    print(f"Algorithms: {SELECTED_ALGORITHMS}")

    # --------------------------------------------------------
    # 1. Resolve experiment paths
    # --------------------------------------------------------

    paths = get_experiment_paths(run_id)

    # --------------------------------------------------------
    # 2. Read detailed trial records
    # --------------------------------------------------------

    print("\n[1/10] Reading trial records...")

    results_df = collect_existing_run_details(
        paths["run_details"]
    )

    # --------------------------------------------------------
    # 3. Validate trial-level records
    # --------------------------------------------------------

    print("\n[2/10] Validating trial records...")

    validate_existing_run_details(
        results_df,
        SELECTED_DATASETS,
        SELECTED_ALGORITHMS,
        EXPECTED_TRIALS,
    )

    # --------------------------------------------------------
    # 4. Read fitness results
    # --------------------------------------------------------

    print("\n[3/10] Reading fitness results...")

    stats_df = read_algorithm_data(
        paths["best_fit"],
        SELECTED_ALGORITHMS,
        expected_trials=EXPECTED_TRIALS,
        expected_datasets=SELECTED_DATASETS,
    )

    fitness_matrix = None

    # Strict completeness validation.
    from helpers import validate_experiment_completeness

    fitness_matrix = validate_experiment_completeness(
        stats_df,
        SELECTED_DATASETS,
        SELECTED_ALGORITHMS,
        EXPECTED_TRIALS,
    )

    print("\nFitness result matrix:")

    print(
        fitness_matrix.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    # --------------------------------------------------------
    # 5. Prepare summaries
    # --------------------------------------------------------

    print("\n[4/10] Preparing result summaries...")

    performance_df = prepare_performance_data(
        stats_df
    )

    test_performance_df = prepare_test_performance_data(
        results_df,
        expected_datasets=SELECTED_DATASETS,
        algorithms=SELECTED_ALGORITHMS,
        expected_trials=EXPECTED_TRIALS,
    )

    runtime_df = prepare_runtime_data(
        results_df,
        expected_datasets=SELECTED_DATASETS,
        algorithms=SELECTED_ALGORITHMS,
        expected_trials=EXPECTED_TRIALS,
    )

    # --------------------------------------------------------
    # 6. Create experiment audit
    # --------------------------------------------------------

    print("\n[5/10] Creating experiment audit...")

    completeness_df = create_experiment_audit(
        stats_df,
        test_performance_df,
        runtime_df,
        SELECTED_DATASETS,
        SELECTED_ALGORITHMS,
        EXPECTED_TRIALS,
    )

    print(
        completeness_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # 7. Statistical tests
    # --------------------------------------------------------

    print("\n[6/10] Running statistical analysis...")

    friedman_results = perform_friedman_test(
        stats_df,
        expected_datasets=SELECTED_DATASETS,
        algorithms=SELECTED_ALGORITHMS,
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

    wilcoxon_results = perform_wilcoxon_test(
        stats_df,
        PROPOSED_ALGORITHM,
        alpha=ALPHA,
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

    nemenyi_results = perform_nemenyi_test(
        friedman_results,
        n_datasets=len(SELECTED_DATASETS),
        alpha=ALPHA,
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

    # --------------------------------------------------------
    # 8. Save Excel report
    # --------------------------------------------------------

    print("\n[7/10] Saving Excel report...")

    save_to_excel(
        performance_df=performance_df,
        test_performance_df=test_performance_df,
        runtime_df=runtime_df,
        friedman_results=friedman_results,
        wilcoxon_results=wilcoxon_results,
        nemenyi_results=nemenyi_results,
        completeness_df=completeness_df,
        filename=paths["excel"],
    )

    print(
        f"Excel report saved:\n{paths['excel']}"
    )

    # --------------------------------------------------------
    # 9. Generate plots
    # --------------------------------------------------------

    print("\n[8/10] Generating convergence plots...")

    convergence_data = read_convergence_data(
        paths["convergence"],
        SELECTED_DATASETS,
        SELECTED_ALGORITHMS,
    )

    plot_convergence_curves(
        convergence_data,
        SELECTED_DATASETS,
        SELECTED_ALGORITHMS,
        paths["plots"],
    )

    print("\nGenerating statistical plots...")

    plot_friedman_ranks(
        friedman_results[2],
        paths["plots"],
    )

    plot_critical_difference(
        friedman_results,
        nemenyi_results,
        paths["plots"],
    )

    plot_wilcoxon_heatmaps(
        wilcoxon_results,
        PROPOSED_ALGORITHM,
        paths["plots"],
    )

    plot_pairwise_win_tie_loss(
        wilcoxon_results,
        proposed_algo=PROPOSED_ALGORITHM,
        save_path=paths["plots"],
    )

    # --------------------------------------------------------
    # 10. Runtime plots and final report
    # --------------------------------------------------------

    print("\n[9/10] Generating runtime plots...")

    plot_runtime_analysis(
        runtime_df,
        results_df,
        SELECTED_DATASETS,
        SELECTED_ALGORITHMS,
        paths["plots"],
    )

    print_runtime_summary(runtime_df)

    print("\n" + "=" * 75)
    print("EXISTING EXPERIMENT ANALYSIS COMPLETED")
    print("=" * 75)

    print(f"Experiment ID: {run_id}")

    print(f"Excel report: {paths['excel']}")

    print(f"Plots folder: {paths['plots']}")

    print(
        f"Total trials analyzed: "
        f"{len(results_df)}"
    )

    print("\nNo optimization algorithms were executed.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Analyze an existing FS-MHA experiment "
            "without rerunning optimization algorithms."
        )
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=DEFAULT_RUN_ID,
        help="Existing experiment ID.",
    )

    args = parser.parse_args()

    try:

        analyze_existing_run(
            run_id=args.run_id
        )

    except Exception as exc:

        print("\n" + "!" * 75)
        print("ANALYSIS FAILED")
        print("!" * 75)

        print(f"{type(exc).__name__}: {exc}")

        sys.exit(1)
