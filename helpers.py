import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
    rankdata,
    studentized_range,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ==========================================================
# Project paths and dataset configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_FOLDER = PROJECT_ROOT / "datasets"

TARGET_COLUMN = "target"

TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20

RANDOM_STATE = 42

ALPHA = 0.05


# ==========================================================
# Dataset utilities
# ==========================================================

def get_available_datasets(dataset_folder=None):

    folder = (
        Path(dataset_folder)
        if dataset_folder is not None
        else DATASET_FOLDER
    )

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder does not exist: {folder}"
        )

    return sorted(
        file.stem
        for file in folder.glob("*.csv")
    )


def _load_csv_dataset(dataset_name, dataset_folder=None):

    folder = (
        Path(dataset_folder)
        if dataset_folder is not None
        else DATASET_FOLDER
    )

    filename = str(dataset_name)

    if filename.lower().endswith(".csv"):
        filename = filename[:-4]

    file_path = folder / f"{filename}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"Dataset is empty: {filename}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found "
            f"in dataset '{filename}'."
        )

    if df.columns.duplicated().any():
        duplicated = (
            df.columns[df.columns.duplicated()].tolist()
        )

        raise ValueError(
            f"Duplicated column names: {duplicated}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Dataset '{filename}' contains missing targets."
        )

    feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]

    if not feature_columns:
        raise ValueError(
            f"Dataset '{filename}' has no features."
        )

    X_df = df[feature_columns]

    non_numeric = X_df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            f"Non-numeric features in '{filename}': {non_numeric}"
        )

    X = X_df.to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(
            f"Dataset '{filename}' contains NaN or infinity."
        )

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Dataset '{filename}' must contain at least "
            "two classes."
        )

    return X, y, feature_columns


# ==========================================================
# Data splitting
# ==========================================================

def _split_without_duplicate_groups(X, y, random_state):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        stratify=y,
        random_state=random_state,
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=random_state,
        )
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        "Stratified",
    )


def _split_with_duplicate_groups(X, y, random_state):

    _, groups = np.unique(
        X,
        axis=0,
        return_inverse=True,
    )

    if len(np.unique(groups)) < 5:
        raise ValueError(
            "Group-aware splitting requires at least "
            "five distinct feature-vector groups."
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    fold_indices = []

    try:
        for _, test_indices in splitter.split(
            X,
            y,
            groups,
        ):
            fold_indices.append(test_indices)

    except ValueError as exc:
        raise ValueError(
            "Group-aware stratified splitting failed. "
            "Check class frequencies and duplicate groups."
        ) from exc

    test_indices = fold_indices[0]
    validation_indices = fold_indices[1]

    train_indices = np.concatenate(
        fold_indices[2:]
    )

    return (
        X[train_indices],
        X[validation_indices],
        X[test_indices],
        y[train_indices],
        y[validation_indices],
        y[test_indices],
        "StratifiedGroupKFold",
    )


def _check_split_classes(
    y_train,
    y_validation,
    y_test,
    dataset_name,
):

    expected_classes = set(np.unique(y_train))

    if len(expected_classes) < 2:
        raise ValueError(
            f"Training partition of '{dataset_name}' "
            "contains fewer than two classes."
        )

    for name, labels in [
        ("Validation", y_validation),
        ("Test", y_test),
    ]:

        missing = (
            expected_classes - set(np.unique(labels))
        )

        if missing:
            raise ValueError(
                f"{name} partition of '{dataset_name}' "
                f"is missing classes: {missing}."
            )


# ==========================================================
# Main dataset converter
# ==========================================================

def convert_datasets(
    datasetName,
    random_state=RANDOM_STATE,
    dataset_folder=None,
):

    X, y, feature_names = _load_csv_dataset(
        datasetName,
        dataset_folder,
    )

    print(f"\nLoading dataset: {datasetName}")
    print(f"Original shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    n_unique_vectors = len(
        np.unique(X, axis=0)
    )

    n_duplicate_vectors = len(X) - n_unique_vectors

    print(
        f"Duplicate feature-vector observations: "
        f"{n_duplicate_vectors}"
    )

    if n_duplicate_vectors > 0:

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
            split_method,
        ) = _split_with_duplicate_groups(
            X,
            y_encoded,
            random_state,
        )

    else:

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
            split_method,
        ) = _split_without_duplicate_groups(
            X,
            y_encoded,
            random_state,
        )

    _check_split_classes(
        y_train,
        y_validation,
        y_test,
        datasetName,
    )

    # Remove constant features using Training only.
    feature_variances = np.var(
        X_train,
        axis=0,
    )

    retained_mask = feature_variances > 0

    removed_features = [
        name
        for name, keep in zip(feature_names, retained_mask)
        if not keep
    ]

    retained_features = [
        name
        for name, keep in zip(feature_names, retained_mask)
        if keep
    ]

    if not np.any(retained_mask):
        raise ValueError(
            f"All features in '{datasetName}' are constant."
        )

    X_train = X_train[:, retained_mask]
    X_validation = X_validation[:, retained_mask]
    X_test = X_test[:, retained_mask]

    # Fit preprocessing on Training only.
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    split_info = {
        "method": split_method,
        "train_samples": len(X_train),
        "validation_samples": len(X_validation),
        "test_samples": len(X_test),
        "train_ratio": len(X_train) / len(X),
        "validation_ratio": len(X_validation) / len(X),
        "test_ratio": len(X_test) / len(X),
    }

    data = {
        "X_train": X_train,
        "y_train": y_train,

        "X_validation": X_validation,
        "y_validation": y_validation,

        "X_test": X_test,
        "y_test": y_test,

        "feature_names": retained_features,
        "original_feature_names": feature_names,
        "removed_constant_features": removed_features,

        "label_encoder": label_encoder,
        "scaler": scaler,

        "split_info": split_info,

        "n_original_features": len(feature_names),
        "n_features": len(retained_features),
        "n_duplicate_vectors": n_duplicate_vectors,
    }

    print(f"Split method: {split_method}")

    print(
        f"Training:   {len(X_train):>7} "
        f"({split_info['train_ratio']:.2%})"
    )

    print(
        f"Validation: {len(X_validation):>7} "
        f"({split_info['validation_ratio']:.2%})"
    )

    print(
        f"Test:       {len(X_test):>7} "
        f"({split_info['test_ratio']:.2%})"
    )

    print(f"Original features: {len(feature_names)}")
    print(f"Retained features: {len(retained_features)}")
    print(f"Removed constant features: {len(removed_features)}")

    if removed_features:
        print(f"Removed: {removed_features}")

    print("Dataset preprocessing completed successfully.")

    return data


# ==========================================================
# Result readers
# ==========================================================

def read_algorithm_data(
    base_folder,
    algorithms,
    expected_trials=None,
    expected_datasets=None,
):
    """
    Read Mealpy best-fitness CSV files.

    Rows = independent trials.
    Columns = datasets.

    Best Trial is one-based and corresponds to the row
    position in the Mealpy CSV file.
    """

    base_folder = Path(base_folder)

    records = []

    for algorithm in algorithms:

        file_path = (
            base_folder / f"{algorithm}_best_fit.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Best-fitness file not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        df = df.loc[
            :,
            ~df.columns.astype(str).str.startswith("Unnamed:")
        ]

        if expected_datasets is not None:

            missing = (
                set(expected_datasets) - set(df.columns)
            )

            if missing:
                raise ValueError(
                    f"{algorithm}: missing datasets in "
                    f"best-fitness CSV: {sorted(missing)}"
                )

        for dataset_name in df.columns:

            values = pd.to_numeric(
                df[dataset_name],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    f"Invalid fitness values in "
                    f"{algorithm}/{dataset_name}."
                )

            costs = values.to_numpy(dtype=float)

            if len(costs) == 0:
                raise ValueError(
                    f"No fitness values for "
                    f"{algorithm}/{dataset_name}."
                )

            if not np.isfinite(costs).all():
                raise ValueError(
                    f"Non-finite fitness values for "
                    f"{algorithm}/{dataset_name}."
                )

            if (
                expected_trials is not None
                and len(costs) != expected_trials
            ):
                raise ValueError(
                    f"{algorithm}/{dataset_name}: expected "
                    f"{expected_trials} trials, found {len(costs)}."
                )

            best_index = int(np.argmin(costs))

            records.append({
                "Function": dataset_name,
                "Algorithm": algorithm,

                "Costs": costs.tolist(),

                "Best Cost": float(costs[best_index]),
                "Best Trial": best_index + 1,

                "Worst Cost": float(np.max(costs)),
                "Average Cost": float(np.mean(costs)),

                "STD": float(
                    np.std(costs, ddof=1)
                    if len(costs) > 1
                    else 0.0
                ),

                "N Trials": len(costs),
            })

    return pd.DataFrame(records)


# ==========================================================
# Test metrics aggregation
# ==========================================================

def prepare_test_performance_data(
    run_details_df,
    expected_datasets=None,
    algorithms=None,
    expected_trials=None,
):
    """
    Aggregate final Test metrics across independent trials.

    Test results are reported separately from fitness.
    They are never used to select the best trial.
    """

    required = {
        "dataset",
        "algorithm",
        "trial",
        "test_metrics",
        "test_evaluation_error",
    }

    if not required.issubset(run_details_df.columns):
        raise ValueError(
            f"Run details must contain: {required}"
        )

    rows = []

    for (dataset, algorithm), group in run_details_df.groupby(
        ["dataset", "algorithm"]
    ):

        if (
            expected_trials is not None
            and len(group) != expected_trials
        ):
            raise ValueError(
                f"{dataset}/{algorithm}: expected "
                f"{expected_trials} records, found {len(group)}."
            )

        errors = group[
            "test_evaluation_error"
        ].dropna()

        if len(errors) > 0:
            raise RuntimeError(
                f"Test evaluation errors in {dataset}/{algorithm}: "
                f"{errors.tolist()}"
            )

        metric_records = []

        for value in group["test_metrics"]:

            if isinstance(value, dict):
                metric_records.append(value)

            elif isinstance(value, str):
                import json
                metric_records.append(json.loads(value))

            else:
                raise ValueError(
                    f"Missing Test metrics in {dataset}/{algorithm}."
                )

        metrics_df = pd.DataFrame(metric_records)

        row = {
            "Dataset": dataset,
            "Algorithm": algorithm,
            "N Trials": len(group),
        }

        metric_columns = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "n_features",
            "total_features",
            "feature_ratio",
        ]

        for metric in metric_columns:

            if metric not in metrics_df.columns:
                continue

            values = pd.to_numeric(
                metrics_df[metric],
                errors="coerce",
            )

            row[f"{metric}_mean"] = values.mean()

            row[f"{metric}_std"] = values.std(
                ddof=1
            ) if len(values) > 1 else 0.0

        rows.append(row)

    result = pd.DataFrame(rows)

    if expected_datasets is not None and algorithms is not None:

        expected_pairs = {
            (dataset, algorithm)
            for dataset in expected_datasets
            for algorithm in algorithms
        }

        actual_pairs = set(
            zip(result["Dataset"], result["Algorithm"])
        )

        missing = expected_pairs - actual_pairs

        if missing:
            raise RuntimeError(
                f"Missing Test evaluation groups: {sorted(missing)}"
            )

    return result.sort_values(
        ["Dataset", "Algorithm"]
    ).reset_index(drop=True)



# ==========================================================
# Runtime analysis
# ==========================================================

def prepare_runtime_data(
    run_details_df,
    expected_datasets=None,
    algorithms=None,
    expected_trials=None,
):
    """
    Aggregate runtime information across independent trials.

    Runtime components:
        1. Optimization time
        2. Test evaluation time
        3. Total trial time

    Total trial time is derived as:
        optimization_time + test_evaluation_time

    This function does not include the overall wall-clock
    duration of the parallel experiment.
    """

    required = {
        "dataset",
        "algorithm",
        "trial",
        "optimization_time",
        "test_evaluation_time",
    }

    if not required.issubset(run_details_df.columns):
        raise ValueError(
            f"Run details must contain: {required}"
        )

    df = run_details_df.copy()

    # ------------------------------------------------------
    # Validate runtime values
    # ------------------------------------------------------

    runtime_columns = [
        "optimization_time",
        "test_evaluation_time",
    ]

    for column in runtime_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_mask = (
            df[column].isna()
            | ~np.isfinite(df[column])
        )

        if invalid_mask.any():
            invalid_records = df.loc[
                invalid_mask,
                ["dataset", "algorithm", "trial", column],
            ]

            raise ValueError(
                f"Invalid values in {column}:\n"
                f"{invalid_records.to_string(index=False)}"
            )

    if (df["optimization_time"] <= 0).any():
        raise ValueError(
            "Optimization time must be greater than zero."
        )

    if (df["test_evaluation_time"] < 0).any():
        raise ValueError(
            "Test evaluation time cannot be negative."
        )

    # ------------------------------------------------------
    # Validate trial identifiers
    # ------------------------------------------------------

    if df.duplicated(
        subset=["dataset", "algorithm", "trial"]
    ).any():

        raise ValueError(
            "Duplicate dataset/algorithm/trial records "
            "were found in runtime data."
        )

    df["total_time"] = (
        df["optimization_time"]
        + df["test_evaluation_time"]
    )

    # ------------------------------------------------------
    # Validate expected groups
    # ------------------------------------------------------

    if expected_datasets is not None and algorithms is not None:

        expected_pairs = {
            (dataset, algorithm)
            for dataset in expected_datasets
            for algorithm in algorithms
        }

        actual_pairs = set(
            zip(df["dataset"], df["algorithm"])
        )

        missing = expected_pairs - actual_pairs

        unexpected = actual_pairs - expected_pairs

        if missing:
            raise RuntimeError(
                f"Missing runtime groups: {sorted(missing)}"
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected runtime groups: {sorted(unexpected)}"
            )

    # ------------------------------------------------------
    # Aggregate runtime statistics
    # ------------------------------------------------------

    rows = []

    for (dataset, algorithm), group in df.groupby(
        ["dataset", "algorithm"]
    ):

        if (
            expected_trials is not None
            and len(group) != expected_trials
        ):
            raise ValueError(
                f"{dataset}/{algorithm}: expected "
                f"{expected_trials} trials, "
                f"found {len(group)}."
            )

        row = {
            "Dataset": dataset,
            "Algorithm": algorithm,
            "N Trials": len(group),
        }

        time_columns = {
            "Optimization": "optimization_time",
            "Test Evaluation": "test_evaluation_time",
            "Total": "total_time",
        }

        for prefix, column in time_columns.items():

            values = group[column]

            row[f"{prefix} Mean (s)"] = values.mean()

            row[f"{prefix} STD (s)"] = (
                values.std(ddof=1)
                if len(values) > 1
                else 0.0
            )

            row[f"{prefix} Median (s)"] = values.median()

            row[f"{prefix} Min (s)"] = values.min()

            row[f"{prefix} Max (s)"] = values.max()

        rows.append(row)

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["Dataset", "Algorithm"]
    ).reset_index(drop=True)



# ==========================================================
# Performance summary
# ==========================================================

def prepare_performance_data(stats_df):

    columns = [
        "Best Cost",
        "Average Cost",
        "STD",
        "Worst Cost",
    ]

    # columns = [
    #     "Best Cost",
    #     "Best Trial",
    #     "Average Cost",
    #     "STD",
    #     "Worst Cost",
    #     "N Trials",
    # ]
    
    if stats_df.empty:
        return pd.DataFrame()

    rows = []

    for dataset in sorted(
        stats_df["Function"].unique()
    ):

        dataset_df = stats_df[
            stats_df["Function"] == dataset
        ]

        for metric in columns:

            row = {
                "Dataset": dataset,
                "Metric": metric,
            }

            for _, entry in dataset_df.iterrows():
                row[entry["Algorithm"]] = entry[metric]

            rows.append(row)

    return (
        pd.DataFrame(rows)
        .set_index(["Dataset", "Metric"])
    )


# ==========================================================
# Paired performance matrix
# ==========================================================

def _build_paired_performance_matrix(stats_summary):

    if isinstance(stats_summary, pd.DataFrame):
        df = stats_summary.copy()

    else:
        df = pd.DataFrame(stats_summary)

    required = {
        "Function",
        "Algorithm",
        "Average Cost",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Statistics data must contain: {required}"
        )

    matrix = df.pivot(
        index="Function",
        columns="Algorithm",
        values="Average Cost",
    )

    return matrix


def validate_experiment_completeness(
    stats_df,
    expected_datasets,
    algorithms,
    expected_trials,
):
    """
    Strict validation: every dataset must have every algorithm,
    and every algorithm/dataset pair must contain the expected
    number of independent trials.
    """

    expected_pairs = {
        (dataset, algorithm)
        for dataset in expected_datasets
        for algorithm in algorithms
    }

    actual_pairs = set(
        zip(stats_df["Function"], stats_df["Algorithm"])
    )

    missing = expected_pairs - actual_pairs
    unexpected = actual_pairs - expected_pairs

    if missing or unexpected:
        raise RuntimeError(
            "Experiment result coverage is incomplete.\n"
            f"Missing pairs: {sorted(missing)}\n"
            f"Unexpected pairs: {sorted(unexpected)}"
        )

    invalid = stats_df[
        stats_df["N Trials"] != expected_trials
    ]

    if not invalid.empty:
        raise RuntimeError(
            "Incorrect trial counts:\n"
            + invalid[
                ["Function", "Algorithm", "N Trials"]
            ].to_string(index=False)
        )

    matrix = _build_paired_performance_matrix(stats_df)

    if matrix.isna().any().any():
        raise RuntimeError(
            "The Friedman/Wilcoxon matrix contains missing values."
        )

    if set(matrix.index) != set(expected_datasets):
        raise RuntimeError(
            "Dataset coverage in statistical matrix is incorrect."
        )

    if set(matrix.columns) != set(algorithms):
        raise RuntimeError(
            "Algorithm coverage in statistical matrix is incorrect."
        )

    return matrix.loc[
        expected_datasets,
        algorithms,
    ]


# ==========================================================
# Friedman test
# ==========================================================

def perform_friedman_test(
    stats_summary,
    expected_datasets=None,
    algorithms=None,
):
    """
    Friedman test on mean fitness values.

    Each dataset contributes one paired observation
    per algorithm.
    """

    matrix = _build_paired_performance_matrix(
        stats_summary
    )

    if expected_datasets is not None and algorithms is not None:

        matrix = matrix.reindex(
            index=expected_datasets,
            columns=algorithms,
        )

        if matrix.isna().any().any():
            raise RuntimeError(
                "Friedman test cannot run: incomplete result matrix."
            )

    else:

        if matrix.isna().any().any():
            raise RuntimeError(
                "Friedman matrix contains missing values. "
                "Do not silently drop incomplete datasets."
            )

    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise ValueError(
            "Friedman test requires at least three datasets "
            "and three algorithms."
        )

    samples = [
        matrix[algorithm].to_numpy(dtype=float)
        for algorithm in matrix.columns
    ]

    statistic, p_value = friedmanchisquare(*samples)

    ranks = matrix.apply(
        lambda row: rankdata(
            row.to_numpy(),
            method="average",
        ),
        axis=1,
        result_type="expand",
    )

    ranks.columns = matrix.columns

    mean_ranks = ranks.mean(axis=0)

    algorithm_ranks = {
        algorithm: float(mean_ranks[algorithm])
        for algorithm in matrix.columns
    }

    return (
        float(statistic),
        float(p_value),
        algorithm_ranks,
    )


# ==========================================================
# Holm correction
# ==========================================================

def holm_correction(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(p_values)
    )[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_indices]

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    m = len(sorted_p)

    adjusted_sorted = np.empty(m)

    running_max = 0.0

    for i, p_value in enumerate(sorted_p):

        corrected = (m - i) * p_value

        running_max = max(
            running_max,
            corrected,
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0,
        )

    adjusted_valid = np.empty(m)
    adjusted_valid[order] = adjusted_sorted

    adjusted[valid_indices] = adjusted_valid

    return adjusted


# ==========================================================
# Wilcoxon signed-rank test with Holm correction
# ==========================================================

def perform_wilcoxon_test(
    stats_summary,
    proposed_algorithm,
    alpha=ALPHA,
):
    """
    Compare proposed algorithm against all competitors.

    Raw p-values are adjusted using Holm's procedure.

    '+' proposed algorithm significantly better.
    '-' proposed algorithm significantly worse.
    '=' no significant difference.
    """

    matrix = _build_paired_performance_matrix(
        stats_summary
    )

    if proposed_algorithm not in matrix.columns:
        raise ValueError(
            f"Proposed algorithm '{proposed_algorithm}' "
            "was not found."
        )

    proposed_values = matrix[proposed_algorithm]

    raw_results = {}

    for algorithm in matrix.columns:

        if algorithm == proposed_algorithm:
            continue

        paired = pd.DataFrame({
            "proposed": proposed_values,
            "competitor": matrix[algorithm],
        }).dropna()

        differences = (
            paired["proposed"].to_numpy()
            - paired["competitor"].to_numpy()
        )

        if len(differences) < 2:

            statistic = np.nan
            p_value = np.nan

        elif np.allclose(differences, 0):

            statistic = 0.0
            p_value = 1.0

        else:

            statistic, p_value = wilcoxon(
                paired["proposed"].to_numpy(),
                paired["competitor"].to_numpy(),
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

        raw_results[algorithm] = {
            "Statistic": float(statistic),
            "Raw p-value": float(p_value),
            "N Datasets": int(len(paired)),
            "Mean Difference": float(np.mean(differences)),
            "Median Difference": float(np.median(differences)),
        }

    p_values = [
        result["Raw p-value"]
        for result in raw_results.values()
    ]

    adjusted_p = holm_correction(p_values)

    final_results = {}

    for index, (algorithm, result) in enumerate(
        raw_results.items()
    ):

        p_adjusted = adjusted_p[index]

        if not np.isfinite(p_adjusted):
            verdict = "N/A"

        elif p_adjusted >= alpha:
            verdict = "="

        elif result["Median Difference"] < 0:
            verdict = "+"

        elif result["Median Difference"] > 0:
            verdict = "-"

        else:
            verdict = "="

        final_results[algorithm] = {
            **result,
            "Holm Adjusted p-value": float(p_adjusted),
            "Better Algorithm": verdict,
        }

    return {
        "Overall": final_results
    }


# ==========================================================
# Nemenyi post-hoc test
# ==========================================================

def perform_nemenyi_test(
    friedman_results,
    n_datasets,
    alpha=ALPHA,
):
    """
    Nemenyi post-hoc comparisons based on Friedman mean ranks.

    The critical difference is calculated using the
    studentized range distribution.
    """

    algorithm_ranks = friedman_results[2]

    algorithms = list(
        algorithm_ranks.keys()
    )

    k = len(algorithms)

    if k < 2 or n_datasets < 2:
        raise ValueError(
            "Nemenyi test requires at least two algorithms "
            "and two datasets."
        )

    q_alpha = (
        studentized_range.ppf(
            1 - alpha,
            k,
            np.inf,
        )
        / np.sqrt(2)
    )

    cd = q_alpha * np.sqrt(
        k * (k + 1) / (6 * n_datasets)
    )

    denominator = np.sqrt(
        k * (k + 1) / (6 * n_datasets)
    )

    rows = []

    for i in range(k):

        for j in range(i + 1, k):

            algo_a = algorithms[i]
            algo_b = algorithms[j]

            rank_a = algorithm_ranks[algo_a]
            rank_b = algorithm_ranks[algo_b]

            difference = abs(
                rank_a - rank_b
            )

            q_statistic = (
                difference / denominator
            )

            p_value = studentized_range.sf(
                q_statistic * np.sqrt(2),
                k,
                np.inf,
            )

            rows.append({
                "Algorithm A": algo_a,
                "Algorithm B": algo_b,
                "Rank A": rank_a,
                "Rank B": rank_b,
                "Absolute Rank Difference": difference,
                "Nemenyi p-value": float(p_value),
                "Significant": bool(difference > cd),
                "Critical Difference": float(cd),
            })

    return {
        "Critical Difference": float(cd),
        "Alpha": alpha,
        "N Datasets": n_datasets,
        "N Algorithms": k,
        "Comparisons": pd.DataFrame(rows),
    }


# ==========================================================
# Convergence data
# ==========================================================

def read_convergence_data(
    base_folder,
    benchmarks,
    algorithms,
):
    """
    Read Mealpy convergence CSV files.

    Confirmed layout:
        Rows = epochs
        Columns = independent trials

    Therefore, average convergence is calculated across
    columns using axis=1.
    """

    convergence_data = {
        benchmark: {}
        for benchmark in benchmarks
    }

    for benchmark in benchmarks:

        for algorithm in algorithms:

            file_path = (
                Path(base_folder)
                / algorithm
                / f"{benchmark}_convergence.csv"
            )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Convergence file not found: {file_path}"
                )

            df = pd.read_csv(file_path)

            df = df.select_dtypes(
                include=[np.number]
            )

            if df.empty:
                raise ValueError(
                    f"Empty convergence data: {file_path}"
                )

            if df.isna().any().any():
                raise ValueError(
                    f"Missing convergence values: {file_path}"
                )

            # Rows are epochs; columns are trials.
            mean_curve = df.mean(axis=1)

            convergence_data[benchmark][algorithm] = (
                mean_curve.reset_index(drop=True)
            )

    return convergence_data


# ==========================================================
# Convergence plots
# ==========================================================

def plot_convergence_curves(
    convergence_data,
    benchmarks,
    algorithms,
    output_folder,
):

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for benchmark in benchmarks:

        plt.figure(figsize=(12, 7))

        plotted = False
        all_values = []

        for algorithm in algorithms:

            curve = convergence_data.get(
                benchmark,
                {},
            ).get(algorithm)

            if curve is None or len(curve) == 0:
                continue

            values = np.asarray(curve, dtype=float)

            plt.plot(
                np.arange(1, len(values) + 1),
                values,
                label=algorithm,
                linewidth=2,
            )

            all_values.extend(values.tolist())
            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.title(
            f"Mean Convergence Curves - {benchmark}"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Mean Fitness")

        if all(value > 0 for value in all_values):
            plt.yscale("log")

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_folder / f"convergence_{benchmark}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ==========================================================
# Friedman rank plot
# ==========================================================

def plot_friedman_ranks(rank_dict, output_folder):

    if not rank_dict:
        warnings.warn("No Friedman ranks available.")
        return

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    sorted_items = sorted(
        rank_dict.items(),
        key=lambda item: item[1],
    )

    algorithms, ranks = zip(*sorted_items)

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=list(algorithms),
        y=list(ranks),
    )

    plt.ylabel("Mean Friedman Rank (Lower is Better)")
    plt.title("Friedman Mean Ranks")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        output_folder / "friedman_ranks_barplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# Critical Difference diagram
# ==========================================================

def plot_critical_difference(
    friedman_results,
    nemenyi_results,
    output_folder,
):
    """
    Draw a rank-axis diagram.

    Algorithms connected by a horizontal line are not
    significantly different according to Nemenyi.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranks = friedman_results[2]

    if not ranks:
        warnings.warn("No ranks available for CD diagram.")
        return

    sorted_items = sorted(
        ranks.items(),
        key=lambda item: item[1],
    )

    algorithms = [
        item[0] for item in sorted_items
    ]

    rank_values = [
        item[1] for item in sorted_items
    ]

    cd = nemenyi_results["Critical Difference"]
    comparisons = nemenyi_results["Comparisons"]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.set_xlim(
        0.5,
        len(algorithms) + 0.5,
    )

    ax.set_ylim(
        -1.5,
        2.5,
    )

    ax.set_yticks([])

    ax.set_xlabel(
        "Mean Friedman Rank (Lower is Better)"
    )

    ax.set_title(
        f"Critical Difference Diagram (CD = {cd:.3f})"
    )

    ax.set_xticks(
        range(1, len(algorithms) + 1)
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    for index, (algorithm, rank) in enumerate(
        zip(algorithms, rank_values)
    ):

        ax.scatter(
            rank,
            0,
            s=80,
            zorder=3,
        )

        label_y = (
            0.35 if index % 2 == 0 else -0.55
        )

        ax.text(
            rank,
            label_y,
            f"{algorithm}\n({rank:.3f})",
            ha="center",
            va="center",
            fontsize=9,
        )

    # Draw non-significant pair connections.
    y_position = 1.2

    for _, row in comparisons.iterrows():

        if not row["Significant"]:

            rank_a = row["Rank A"]
            rank_b = row["Rank B"]

            left = min(rank_a, rank_b)
            right = max(rank_a, rank_b)

            ax.plot(
                [left, right],
                [y_position, y_position],
                linewidth=3,
            )

            y_position -= 0.18

    ax.text(
        0.02,
        0.02,
        "Horizontal bars indicate non-significant "
        "Nemenyi comparisons.",
        transform=ax.transAxes,
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        output_folder / "critical_difference_diagram.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# Wilcoxon plots
# ==========================================================

def plot_wilcoxon_heatmaps(
    wilcoxon_results,
    proposed_algo,
    output_folder,
):

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = wilcoxon_results.get(
        "Overall",
        wilcoxon_results,
    )

    if not results:
        return

    metrics = [
        ("Statistic", "Wilcoxon Statistic"),
        ("Raw p-value", "Wilcoxon Raw p-values"),
        ("Holm Adjusted p-value", "Holm Adjusted p-values"),
    ]

    for key, title in metrics:

        values = {
            f"{proposed_algo} vs {algorithm}": (
                result.get(key, np.nan)
            )
            for algorithm, result in results.items()
        }

        df = pd.DataFrame(
            [values],
            index=["Overall"],
        )

        plt.figure(
            figsize=(max(8, len(df.columns) * 1.7), 3)
        )

        sns.heatmap(
            df,
            annot=True,
            fmt=".3g",
            cmap="coolwarm",
            cbar=True,
        )

        plt.title(title)
        plt.xlabel("Algorithm Comparison")
        plt.tight_layout()

        filename = (
            key.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_heatmap.png"
        )

        plt.savefig(
            output_folder / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ==========================================================
# Win / Tie / Loss summary
# ==========================================================

def plot_pairwise_win_tie_loss(
    wilcoxon_results,
    proposed_algo,
    save_path,
):

    results = wilcoxon_results.get(
        "Overall",
        wilcoxon_results,
    )

    if not results:
        return

    labels = list(results.keys())

    wins = []
    ties = []
    losses = []

    for algorithm in labels:

        verdict = results[algorithm]["Better Algorithm"]

        wins.append(1 if verdict == "+" else 0)
        ties.append(1 if verdict == "=" else 0)
        losses.append(1 if verdict == "-" else 0)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x,
        wins,
        label="Win",
    )

    ax.bar(
        x,
        ties,
        bottom=wins,
        label="Tie",
    )

    bottoms = [
        wins[i] + ties[i]
        for i in range(len(labels))
    ]

    ax.bar(
        x,
        losses,
        bottom=bottoms,
        label="Loss",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Number of Pairwise Comparisons")
    ax.set_title(
        f"{proposed_algo}: Win / Tie / Loss"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path / "win_tie_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# Excel reporting
# ==========================================================

def save_to_excel(
    performance_df,
    test_performance_df,
    runtime_df,
    friedman_results,
    wilcoxon_results,
    nemenyi_results,
    completeness_df,
    filename,
):

    output_path = Path(filename)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistic, p_value, ranks = friedman_results

    friedman_df = pd.DataFrame([{
        "Friedman Statistic": statistic,
        "p-value": p_value,
        "N Datasets": nemenyi_results["N Datasets"],
        "N Algorithms": nemenyi_results["N Algorithms"],
        "Critical Difference": nemenyi_results[
            "Critical Difference"
        ],
        "Alpha": nemenyi_results["Alpha"],
    }])

    ranks_df = pd.DataFrame(
        [
            {
                "Algorithm": algorithm,
                "Mean Rank": rank,
            }
            for algorithm, rank in ranks.items()
        ]
    ).sort_values("Mean Rank")

    wilcoxon_df = pd.DataFrame.from_dict(
        wilcoxon_results["Overall"],
        orient="index",
    )

    wilcoxon_df.index.name = "Competitor"

    nemenyi_df = nemenyi_results["Comparisons"]

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        performance_df.to_excel(
            writer,
            sheet_name="Fitness Summary",
        )

        test_performance_df.to_excel(
            writer,
            sheet_name="Test Performance",
            index=False,
        )
        
        runtime_df.to_excel(
            writer,
            sheet_name="Runtime Analysis",
            index=False,
        )

        friedman_df.to_excel(
            writer,
            sheet_name="Friedman",
            index=False,
        )

        ranks_df.to_excel(
            writer,
            sheet_name="Friedman Ranks",
            index=False,
        )

        wilcoxon_df.to_excel(
            writer,
            sheet_name="Wilcoxon-Holm",
        )

        nemenyi_df.to_excel(
            writer,
            sheet_name="Nemenyi-CD",
            index=False,
        )

        completeness_df.to_excel(
            writer,
            sheet_name="Experiment Audit",
            index=False,
        )

        # Basic worksheet formatting.
        for worksheet in writer.sheets.values():

            worksheet.freeze_panes = "A2"

            for column_cells in worksheet.columns:

                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:

                    if cell.value is not None:
                        max_length = max(
                            max_length,
                            len(str(cell.value)),
                        )

                worksheet.column_dimensions[
                    column_letter
                ].width = min(max_length + 2, 35)


# ==========================================================
# Legacy metrics summary
# ==========================================================

def summarize_metrics(metrics_root="history/metrics"):

    root = Path(metrics_root)

    if not root.exists():
        warnings.warn(
            f"Metrics directory does not exist: {root}"
        )
        return pd.DataFrame()

    summary_rows = []

    for algorithm_folder in sorted(root.iterdir()):

        if not algorithm_folder.is_dir():
            continue

        algorithm_name = algorithm_folder.name
        dataset_trials = {}

        for file_path in sorted(
            algorithm_folder.glob("*.csv")
        ):

            try:

                df = pd.read_csv(file_path)

                if (
                    df.empty
                    or "fitness_value" not in df.columns
                ):
                    continue

                best_row = df.loc[
                    df["fitness_value"].idxmin()
                ]

                dataset_name = file_path.stem

                dataset_trials.setdefault(
                    dataset_name,
                    [],
                ).append(best_row.to_dict())

            except Exception as exc:

                warnings.warn(
                    f"Could not process {file_path}: {exc}"
                )

        for dataset_name, trials in dataset_trials.items():

            trial_df = pd.DataFrame(trials)

            row = {
                "dataset": dataset_name,
                "algorithm": algorithm_name,
                "n_trials": len(trial_df),
            }

            for metric in [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "n_features",
                "feature_ratio",
                "fitness_value",
            ]:

                if metric not in trial_df.columns:
                    continue

                values = pd.to_numeric(
                    trial_df[metric],
                    errors="coerce",
                )

                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(
                    ddof=1
                )

            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ==========================================================
# Dataset smoke test
# ==========================================================

if __name__ == "__main__":

    print("\nHelpers Full Dataset Smoke Test")
    print("=" * 65)

    datasets = get_available_datasets()

    successful = []
    failed = []

    for index, dataset_name in enumerate(
        datasets,
        start=1,
    ):

        print(
            f"\n[{index}/{len(datasets)}] "
            f"Testing: {dataset_name}"
        )

        try:

            data = convert_datasets(dataset_name)

            required_keys = [
                "X_train",
                "y_train",
                "X_validation",
                "y_validation",
                "X_test",
                "y_test",
            ]

            for key in required_keys:
                assert key in data

            n_features = data["X_train"].shape[1]

            assert n_features > 0

            for partition in [
                "train",
                "validation",
                "test",
            ]:

                X_part = data[f"X_{partition}"]
                y_part = data[f"y_{partition}"]

                assert len(X_part) == len(y_part)
                assert len(X_part) > 0

            for key in [
                "X_train",
                "X_validation",
                "X_test",
            ]:
                assert np.isfinite(data[key]).all()

            expected_classes = set(
                np.unique(data["y_train"])
            )

            assert expected_classes == set(
                np.unique(data["y_validation"])
            )

            assert expected_classes == set(
                np.unique(data["y_test"])
            )

            successful.append(dataset_name)

            print(
                f"PASSED | Features: {n_features}"
            )

        except Exception as exc:

            failed.append({
                "dataset": dataset_name,
                "error": str(exc),
            })

            print(f"FAILED: {exc}")

    print("\n" + "=" * 65)
    print("FINAL REPORT")
    print("=" * 65)

    print(f"Total: {len(datasets)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    for item in failed:
        print(
            f"[ERROR] {item['dataset']}: {item['error']}"
        )

    if not failed:
        print("All dataset tests passed.")
        


# ==========================================================
# Runtime plots
# ==========================================================

def plot_runtime_analysis(
    runtime_df,
    run_details_df,
    datasets,
    algorithms,
    output_folder,
):
    """
    Generate runtime comparison plots:

        1. Optimization runtime with standard deviation.
        2. Optimization runtime boxplot across trials.
        3. Optimization runtime heatmap.

    All times are reported in seconds.
    """

    import matplotlib.pyplot as plt

    output_path = Path(output_folder)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Prepare mean and standard deviation matrices
    # ------------------------------------------------------

    mean_matrix = runtime_df.pivot(
        index="Dataset",
        columns="Algorithm",
        values="Optimization Mean (s)",
    ).reindex(
        index=datasets,
        columns=algorithms,
    )

    std_matrix = runtime_df.pivot(
        index="Dataset",
        columns="Algorithm",
        values="Optimization STD (s)",
    ).reindex(
        index=datasets,
        columns=algorithms,
    )

    if mean_matrix.isna().any().any():
        raise ValueError(
            "Missing optimization runtime values "
            "in the plotting matrix."
        )

    # ------------------------------------------------------
    # 1. Grouped bar chart with standard deviation
    # ------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(datasets))
    width = 0.8 / len(algorithms)

    for index, algorithm in enumerate(algorithms):

        positions = (
            x - 0.4
            + width / 2
            + index * width
        )

        ax.bar(
            positions,
            mean_matrix[algorithm].values,
            width,
            yerr=std_matrix[algorithm].values,
            capsize=4,
            label=algorithm,
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Optimization Time (seconds)")
    ax.set_title("Mean Optimization Runtime")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(title="Algorithm")
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    fig.tight_layout()

    fig.savefig(
        output_path / "runtime_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------
    # 2. Boxplot across independent trials
    # ------------------------------------------------------

    fig, ax = plt.subplots(figsize=(13, 6))

    box_data = []
    box_labels = []

    for dataset in datasets:

        for algorithm in algorithms:

            values = run_details_df.loc[
                (
                    run_details_df["dataset"] == dataset
                )
                & (
                    run_details_df["algorithm"] == algorithm
                ),
                "optimization_time",
            ]

            if values.empty:
                raise ValueError(
                    f"No runtime observations for "
                    f"{dataset}/{algorithm}."
                )

            box_data.append(values.to_numpy())

            box_labels.append(
                f"{dataset}\n{algorithm}"
            )

    ax.boxplot(
        box_data,
        labels=box_labels,
        showmeans=True,
    )

    ax.set_xlabel("Dataset / Algorithm")
    ax.set_ylabel("Optimization Time (seconds)")
    ax.set_title(
        "Optimization Runtime Distribution Across Trials"
    )

    ax.tick_params(
        axis="x",
        labelrotation=0,
        labelsize=8,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    fig.tight_layout()

    fig.savefig(
        output_path / "runtime_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------
    # 3. Runtime heatmap
    # ------------------------------------------------------

    values = mean_matrix.to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))

    image = ax.imshow(
        values,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(algorithms))
    )

    ax.set_xticklabels(algorithms)

    ax.set_yticks(
        np.arange(len(datasets))
    )

    ax.set_yticklabels(datasets)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Dataset")
    ax.set_title(
        "Mean Optimization Runtime Heatmap (seconds)"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label("Seconds")

    for i in range(len(datasets)):

        for j in range(len(algorithms)):

            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        output_path / "runtime_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
