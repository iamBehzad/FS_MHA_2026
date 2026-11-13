
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
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


# ==========================================================
# Dataset utilities
# ==========================================================

def get_available_datasets(dataset_folder=None):
    """
    Return the names of available CSV datasets.

    The target column must be named 'target'.
    """
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
    """
    Load one CSV dataset and validate its basic structure.
    """
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
        duplicated_columns = (
            df.columns[df.columns.duplicated()].tolist()
        )

        raise ValueError(
            f"Duplicated column names: {duplicated_columns}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Dataset '{filename}' contains missing target values."
        )

    feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]

    if not feature_columns:
        raise ValueError(
            f"Dataset '{filename}' has no feature columns."
        )

    X_df = df[feature_columns]

    non_numeric_columns = X_df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric_columns:
        raise TypeError(
            f"Dataset '{filename}' contains non-numeric "
            f"features: {non_numeric_columns}"
        )

    X = X_df.to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(
            f"Dataset '{filename}' contains NaN or infinite "
            f"feature values. Clean the source dataset first."
        )

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Dataset '{filename}' must contain at least "
            f"two classes."
        )

    return X, y, feature_columns


# ==========================================================
# Data splitting
# ==========================================================

def _split_without_duplicate_groups(X, y, random_state):
    """
    Stratified 60/20/20 split for datasets without
    duplicated feature vectors.
    """
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
    """
    Group-aware stratified split.

    Identical feature vectors are assigned to the same
    partition to prevent identical observations from
    appearing in both training and evaluation partitions.

    Five folds are used:
        3 folds -> Training
        1 fold  -> Validation
        1 fold  -> Test

    The resulting proportions are approximately 60/20/20.
    """
    _, groups = np.unique(
        X,
        axis=0,
        return_inverse=True,
    )

    n_groups = len(np.unique(groups))

    if n_groups < 5:
        raise ValueError(
            "Group-aware splitting requires at least five "
            "distinct feature-vector groups."
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    fold_indices = []

    try:
        for _, fold_indices_test in splitter.split(
            X,
            y,
            groups,
        ):
            fold_indices.append(fold_indices_test)

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

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_validation = X[validation_indices]
    y_validation = y[validation_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        "StratifiedGroupKFold",
    )


def _check_split_classes(y_train, y_validation, y_test, dataset_name):
    """
    Ensure every class is represented in all three partitions.
    """
    expected_classes = set(np.unique(y_train))

    if len(expected_classes) < 2:
        raise ValueError(
            f"Training partition of '{dataset_name}' "
            "contains fewer than two classes."
        )

    for partition_name, labels in [
        ("Validation", y_validation),
        ("Test", y_test),
    ]:
        partition_classes = set(np.unique(labels))

        missing_classes = expected_classes - partition_classes

        if missing_classes:
            raise ValueError(
                f"{partition_name} partition of '{dataset_name}' "
                f"is missing classes: {missing_classes}. "
                "The dataset may have too few samples per class "
                "or highly concentrated duplicate groups."
            )


# ==========================================================
# Main dataset converter
# ==========================================================

def convert_datasets(
    datasetName,
    random_state=RANDOM_STATE,
    dataset_folder=None,
):
    """
    Load and preprocess a CSV dataset for binary feature selection.

    Processing pipeline:
        1. Load CSV
        2. Validate data
        3. Encode target labels
        4. Detect duplicate feature vectors
        5. Split into Training / Validation / Test
        6. Remove constant features based on Training
        7. Fit MinMaxScaler on Training only
        8. Transform Validation and Test

    Returns:
        Dictionary compatible with MyProblem.py.
    """
    X, y, feature_names = _load_csv_dataset(
        datasetName,
        dataset_folder,
    )

    print(f"\nLoading dataset: {datasetName}")
    print(f"Original shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # ------------------------------------------------------
    # 1. Encode target labels
    # ------------------------------------------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # ------------------------------------------------------
    # 2. Detect duplicate feature vectors
    # ------------------------------------------------------
    n_unique_vectors = len(
        np.unique(X, axis=0)
    )

    n_duplicate_vectors = len(X) - n_unique_vectors

    has_duplicate_vectors = n_duplicate_vectors > 0

    print(
        f"Duplicate feature-vector observations: "
        f"{n_duplicate_vectors}"
    )

    # ------------------------------------------------------
    # 3. Split data
    # ------------------------------------------------------
    if has_duplicate_vectors:
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

    # ------------------------------------------------------
    # 4. Remove constant features using Training only
    # ------------------------------------------------------
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
            f"All features in '{datasetName}' are constant "
            "in the Training partition."
        )

    X_train = X_train[:, retained_mask]
    X_validation = X_validation[:, retained_mask]
    X_test = X_test[:, retained_mask]

    # ------------------------------------------------------
    # 5. Fit scaler on Training only
    # ------------------------------------------------------
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_validation = scaler.transform(X_validation)

    X_test = scaler.transform(X_test)

    # ------------------------------------------------------
    # 6. Prepare metadata
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # 7. Report
    # ------------------------------------------------------
    print(f"Split method: {split_method}")

    print(
        f"Training:   {X_train.shape[0]:>7} samples "
        f"({split_info['train_ratio']:.2%})"
    )

    print(
        f"Validation: {X_validation.shape[0]:>7} samples "
        f"({split_info['validation_ratio']:.2%})"
    )

    print(
        f"Test:       {X_test.shape[0]:>7} samples "
        f"({split_info['test_ratio']:.2%})"
    )

    print(
        f"Original features: {len(feature_names)}"
    )

    print(
        f"Retained features: {len(retained_features)}"
    )

    print(
        f"Removed constant features: {len(removed_features)}"
    )

    if removed_features:
        print(
            f"Removed: {removed_features}"
        )

    print("Dataset preprocessing completed successfully.")

    return data


# ==========================================================
# Legacy converter
# ==========================================================

def convert_datasets2(datasetName):
    """
    Legacy compatibility function.

    The new project uses CSV datasets.
    Biological MAT dataset loading is no longer supported.
    """
    warnings.warn(
        "convert_datasets2() is deprecated. "
        "Use convert_datasets() with a CSV dataset instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return convert_datasets(datasetName)


# ==========================================================
# Algorithm result readers
# ==========================================================

def read_algorithm_data(base_folder, algorithms):
    """
    Read best-fitness CSV files.

    Expected format:
        One column per dataset.
        Rows represent independent trials.

    Returns one record per algorithm and dataset.
    """
    stats_summary = []

    for algorithm in algorithms:
        file_path = Path(base_folder) / (
            f"{algorithm}_best_fit.csv"
        )

        if not file_path.exists():
            warnings.warn(
                f"Best-fitness file not found: {file_path}"
            )
            continue

        df = pd.read_csv(file_path)

        # Remove accidental index columns.
        df = df.loc[
            :,
            ~df.columns.astype(str).str.startswith("Unnamed:")
        ]

        for dataset_name in df.columns:
            costs = pd.to_numeric(
                df[dataset_name],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)

            if len(costs) == 0:
                continue

            stats_summary.append({
                "Function": dataset_name,
                "Algorithm": algorithm,
                "Costs": costs.tolist(),
                "Best Cost": float(np.min(costs)),
                "Worst Cost": float(np.max(costs)),
                "Average Cost": float(np.mean(costs)),
                "STD": float(
                    np.std(costs, ddof=1)
                    if len(costs) > 1
                    else 0.0
                ),
                "N Trials": len(costs),
            })

    return pd.DataFrame(stats_summary)


def read_convergence_data(base_folder, benchmarks, algorithms):
    """
    Read convergence curves and calculate mean convergence
    across independent trials.
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
                convergence_data[benchmark][algorithm] = None
                continue

            df = pd.read_csv(file_path)

            df = df.select_dtypes(
                include=[np.number]
            )

            if df.empty:
                convergence_data[benchmark][algorithm] = None
                continue

            convergence_data[benchmark][algorithm] = (
                df.mean(axis=1)
            )

    return convergence_data


# ==========================================================
# Performance summary
# ==========================================================

def prepare_performance_data(stats_df):
    """
    Prepare a MultiIndex performance table.

    Friedman and Wilcoxon tests use per-dataset average costs.
    """
    columns = [
        "Best Cost",
        "Average Cost",
        "STD",
    ]

    if stats_df.empty:
        return pd.DataFrame(
            columns=columns,
            index=pd.MultiIndex.from_arrays(
                [[], []],
                names=["Dataset", "Metric"],
            ),
        )

    rows = []

    for dataset_name in sorted(
        stats_df["Function"].unique()
    ):
        dataset_df = stats_df[
            stats_df["Function"] == dataset_name
        ]

        for metric in columns:
            row = {
                "Dataset": dataset_name,
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
# Statistical tests
# ==========================================================

def _build_paired_performance_matrix(stats_summary):
    """
    Construct a dataset-by-algorithm matrix using mean
    fitness values across independent trials.
    """
    if isinstance(stats_summary, pd.DataFrame):
        df = stats_summary.copy()

    else:
        df = pd.DataFrame(stats_summary)

    required_columns = {
        "Function",
        "Algorithm",
        "Average Cost",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Statistics data must contain: {required_columns}"
        )

    return df.pivot_table(
        index="Function",
        columns="Algorithm",
        values="Average Cost",
        aggfunc="mean",
    )


def perform_friedman_test(stats_summary):
    """
    Friedman test across algorithms.

    Each dataset contributes one paired observation
    per algorithm: its mean fitness across trials.

    Lower fitness is better.
    """
    matrix = _build_paired_performance_matrix(
        stats_summary
    )

    matrix = matrix.dropna(
        axis=0,
        how="any",
    )

    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        warnings.warn(
            "Friedman test requires at least three datasets "
            "and three algorithms with complete paired results."
        )

        return np.nan, np.nan, {}

    algorithm_names = matrix.columns.tolist()

    samples = [
        matrix[algorithm].to_numpy()
        for algorithm in algorithm_names
    ]

    statistic, p_value = friedmanchisquare(*samples)

    # Calculate mean ranks for each algorithm.
    ranks = matrix.apply(
        lambda row: rankdata(
            row.to_numpy(),
            method="average",
        ),
        axis=1,
        result_type="expand",
    )

    ranks.columns = algorithm_names

    mean_ranks = ranks.mean(axis=0)

    algorithm_ranks = {
        algorithm: float(mean_ranks[algorithm])
        for algorithm in algorithm_names
    }

    return (
        float(statistic),
        float(p_value),
        algorithm_ranks,
    )


def perform_wilcoxon_test(
    stats_summary,
    proposed_algorithm,
):
    """
    Pairwise Wilcoxon signed-rank tests.

    Each pair is compared across datasets using the mean
    fitness of each algorithm over its independent trials.

    Result symbols:
        + : Proposed algorithm is significantly better.
        - : Proposed algorithm is significantly worse.
        = : No statistically significant difference.

    Since this is a minimization problem, lower fitness
    indicates better performance.
    """
    matrix = _build_paired_performance_matrix(
        stats_summary
    )

    if proposed_algorithm not in matrix.columns:
        raise ValueError(
            f"Proposed algorithm '{proposed_algorithm}' "
            "was not found in the result data."
        )

    proposed_values = matrix[proposed_algorithm]

    wilcoxon_results = {}

    for algorithm in matrix.columns:

        if algorithm == proposed_algorithm:
            continue

        paired = pd.DataFrame({
            "proposed": proposed_values,
            "competitor": matrix[algorithm],
        }).dropna()

        if len(paired) < 2:
            statistic = np.nan
            p_value = np.nan
            verdict = "N/A"

        else:
            differences = (
                paired["proposed"].to_numpy()
                - paired["competitor"].to_numpy()
            )

            if np.allclose(differences, 0):
                statistic = 0.0
                p_value = 1.0
                verdict = "="

            else:
                statistic, p_value = wilcoxon(
                    paired["proposed"].to_numpy(),
                    paired["competitor"].to_numpy(),
                    alternative="two-sided",
                )

                if p_value >= 0.05:
                    verdict = "="

                elif (
                    paired["proposed"].mean()
                    < paired["competitor"].mean()
                ):
                    verdict = "+"

                else:
                    verdict = "-"

        wilcoxon_results[algorithm] = {
            "Statistic": float(statistic),
            "p-value": float(p_value),
            "Better Algorithm": verdict,
            "N Datasets": len(paired),
        }

    return {
        "Overall": wilcoxon_results
    }


# ==========================================================
# Convergence plots
# ==========================================================

def plot_convergence_curves(
    convergence_data,
    benchmarks,
    algorithms,
    output_folder,
):
    """
    Plot average convergence curves for all datasets.
    """
    os.makedirs(output_folder, exist_ok=True)

    for benchmark in benchmarks:

        plt.figure(figsize=(12, 7))

        plotted = False

        for algorithm in algorithms:

            curve = convergence_data.get(
                benchmark,
                {},
            ).get(algorithm)

            if curve is None or curve.empty:
                continue

            plt.plot(
                curve.to_numpy(),
                label=algorithm,
                linewidth=2,
            )

            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.title(
            f"Convergence Curves - {benchmark}"
        )

        plt.xlabel("Iterations")
        plt.ylabel("Fitness")

        # Use logarithmic scale only when all values are positive.
        all_values = []

        for algorithm in algorithms:
            curve = convergence_data.get(
                benchmark,
                {},
            ).get(algorithm)

            if curve is not None and not curve.empty:
                all_values.extend(curve.to_numpy())

        if all(value > 0 for value in all_values):
            plt.yscale("log")

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_filename = (
            Path(output_folder)
            / f"convergence_curve_{benchmark}.png"
        )

        plt.savefig(
            output_filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ==========================================================
# Excel reporting
# ==========================================================

def save_to_excel(
    performance_df,
    friedman_results,
    wilcoxon_results,
    filename,
):
    """
    Save performance, Friedman and Wilcoxon results.
    """
    output_path = Path(filename)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(output_path) as writer:

        if not performance_df.empty:
            performance_df.to_excel(
                writer,
                sheet_name="Performance",
            )

        statistic, p_value, algorithm_ranks = (
            friedman_results
        )

        friedman_df = pd.DataFrame({
            "Statistic": [statistic],
            "p-value": [p_value],
        })

        friedman_df.to_excel(
            writer,
            sheet_name="Friedman Test",
            index=False,
        )

        ranks_df = pd.DataFrame(
            list(algorithm_ranks.items()),
            columns=["Algorithm", "Mean Rank"],
        )

        ranks_df.to_excel(
            writer,
            sheet_name="Friedman Ranks",
            index=False,
        )

        if wilcoxon_results:

            if "Overall" in wilcoxon_results:
                comparison_results = wilcoxon_results["Overall"]
            else:
                comparison_results = wilcoxon_results

            wilcoxon_df = pd.DataFrame.from_dict(
                comparison_results,
                orient="index",
            )

            wilcoxon_df.index.name = "Algorithm"

            wilcoxon_df.to_excel(
                writer,
                sheet_name="Wilcoxon",
            )


# ==========================================================
# Wilcoxon heatmaps
# ==========================================================

def plot_wilcoxon_heatmaps(
    wilcoxon_results,
    proposed_algo,
    output_folder,
):
    """
    Plot pairwise Wilcoxon comparison heatmaps.
    """
    os.makedirs(output_folder, exist_ok=True)

    if "Overall" in wilcoxon_results:
        wilcoxon_results = wilcoxon_results["Overall"]

    if not wilcoxon_results:
        warnings.warn(
            "No Wilcoxon results available for plotting."
        )
        return

    metrics = [
        ("Statistic", "Wilcoxon Statistic"),
        ("p-value", "Wilcoxon p-value"),
        ("Better Algorithm", "Wilcoxon Result"),
    ]

    for metric_key, title in metrics:

        values = {}

        for algorithm, result in wilcoxon_results.items():

            value = result.get(metric_key)

            if metric_key == "Better Algorithm":
                value = {
                    "+": 1,
                    "=": 0,
                    "-": -1,
                    "N/A": np.nan,
                }.get(value, np.nan)

            values[f"{proposed_algo} vs {algorithm}"] = value

        df = pd.DataFrame(
            [values],
            index=["Overall"],
        )

        if df.empty or df.isna().all().all():
            continue

        plt.figure(
            figsize=(max(8, len(df.columns) * 1.5), 3)
        )

        sns.heatmap(
            df,
            annot=True,
            fmt=".2f" if metric_key != "Better Algorithm" else ".0f",
            cmap=(
                "coolwarm"
                if metric_key != "Better Algorithm"
                else "bwr"
            ),
            center=0 if metric_key == "Better Algorithm" else None,
            cbar=True,
        )

        plt.title(title)
        plt.xlabel("Algorithm Comparison")
        plt.ylabel("")
        plt.tight_layout()

        filename = (
            f"wilcoxon_{metric_key.replace(' ', '_').lower()}"
            "_heatmap.png"
        )

        plt.savefig(
            os.path.join(output_folder, filename),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ==========================================================
# Friedman rank plot
# ==========================================================

def plot_friedman_ranks(rank_dict, output_path):
    """
    Plot mean Friedman ranks.
    Lower rank is better.
    """
    if not rank_dict:
        warnings.warn(
            "No Friedman ranks available for plotting."
        )
        return

    os.makedirs(output_path, exist_ok=True)

    sorted_algorithms = sorted(
        rank_dict.items(),
        key=lambda item: item[1],
    )

    algorithms, ranks = zip(*sorted_algorithms)

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=list(algorithms),
        y=list(ranks),
    )

    plt.ylabel("Mean Rank (Lower is Better)")
    plt.title("Friedman Mean Ranks")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_path,
            "friedman_ranks_barplot.png",
        ),
        dpi=300,
    )

    plt.close()


# ==========================================================
# Win / Tie / Loss plot
# ==========================================================

def plot_pairwise_win_tie_loss(
    wilcoxon_results,
    proposed_algo="ARBOS",
    save_path=None,
):
    """
    Plot significant wins, ties and losses of the proposed
    algorithm against competitors.

    + : Proposed algorithm significantly better.
    - : Proposed algorithm significantly worse.
    = : No significant difference.
    """
    if "Overall" in wilcoxon_results:
        results = wilcoxon_results["Overall"]
    else:
        results = wilcoxon_results

    if not results:
        warnings.warn(
            "No pairwise results available."
        )
        return

    labels = sorted(results.keys())

    wins = []
    ties = []
    losses = []

    for algorithm in labels:

        verdict = results[algorithm].get(
            "Better Algorithm",
            "N/A",
        )

        wins.append(1 if verdict == "+" else 0)
        ties.append(1 if verdict == "=" else 0)
        losses.append(1 if verdict == "-" else 0)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x,
        wins,
        label="Win",
        color="tab:blue",
    )

    ax.bar(
        x,
        ties,
        bottom=wins,
        label="Tie",
        color="tab:gray",
    )

    bottom_losses = [
        w + t
        for w, t in zip(wins, ties)
    ]

    ax.bar(
        x,
        losses,
        bottom=bottom_losses,
        label="Loss",
        color="tab:red",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Number of Comparisons")
    ax.set_title(
        f"Win-Tie-Loss: {proposed_algo} vs Competitors"
    )

    ax.legend()
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        output_file = os.path.join(
            save_path,
            f"win_tie_loss_{proposed_algo.lower()}.png",
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    else:
        plt.show()

    plt.close()


# ==========================================================
# Metrics summary
# ==========================================================

def summarize_metrics(metrics_root="history/metrics"):
    """
    Summarize trial-level metric CSV files.

    Expected directory structure:

        metrics/
            AlgorithmName/
                trial_*.csv

    Each trial file should contain at least:
        fitness_value
        n_features

    Other metric columns are included when available.
    """
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

                if df.empty or "fitness_value" not in df.columns:
                    continue

                best_row = df.loc[
                    df["fitness_value"].idxmin()
                ]

                # Dataset name is derived from the file name.
                # This parser supports filenames containing underscores.
                parts = file_path.stem.split("_")

                if len(parts) >= 3 and parts[0].lower() in {
                    "trial",
                    "run",
                    "result",
                }:
                    dataset_name = "_".join(parts[1:-1])
                else:
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

            df_trials = pd.DataFrame(trials)

            row = {
                "dataset": dataset_name,
                "algorithm": algorithm_name,
                "n_trials": len(df_trials),
            }

            metric_columns = [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "n_features",
                "feature_ratio",
                "classification_error",
                "error",
                "fitness_value",
            ]

            for metric in metric_columns:

                if metric not in df_trials.columns:
                    continue

                values = pd.to_numeric(
                    df_trials[metric],
                    errors="coerce",
                )

                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(
                    ddof=1
                )

            summary_rows.append(row)

    if not summary_rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)

    return (
        summary_df
        .sort_values(
            by=["dataset", "algorithm"]
        )
        .reset_index(drop=True)
    )


# ==========================================================
# Smoke test
# ==========================================================


if __name__ == "__main__":

    print("\nHelpers Full Dataset Smoke Test")
    print("=" * 65)

    datasets = get_available_datasets()

    print(f"Available datasets: {len(datasets)}")

    successful = []
    failed = []

    for index, dataset_name in enumerate(datasets, start=1):

        print("\n" + "-" * 65)
        print(f"[{index}/{len(datasets)}] Testing: {dataset_name}")
        print("-" * 65)

        try:
            data = convert_datasets(
                dataset_name,
                random_state=RANDOM_STATE,
            )

            # Validate required keys
            required_keys = [
                "X_train",
                "y_train",
                "X_validation",
                "y_validation",
                "X_test",
                "y_test",
            ]

            for key in required_keys:
                assert key in data, f"Missing key: {key}"

            # Validate feature dimensions
            n_features = data["X_train"].shape[1]

            assert n_features > 0

            assert (
                data["X_validation"].shape[1] == n_features
            )

            assert (
                data["X_test"].shape[1] == n_features
            )

            # Validate sample counts
            for partition in ["train", "validation", "test"]:
                X_part = data[
                    "X_train" if partition == "train"
                    else f"X_{partition}"
                ]

                y_part = data[
                    "y_train" if partition == "train"
                    else f"y_{partition}"
                ]

                assert len(X_part) == len(y_part)
                assert len(X_part) > 0

            # Validate numerical values
            for key in [
                "X_train",
                "X_validation",
                "X_test",
            ]:
                assert np.isfinite(data[key]).all()

            # Validate class coverage
            train_classes = set(
                np.unique(data["y_train"])
            )

            assert train_classes == set(
                np.unique(data["y_validation"])
            )

            assert train_classes == set(
                np.unique(data["y_test"])
            )

            successful.append(dataset_name)

            print(
                f"PASSED | Features: {n_features} | "
                f"Split: {data['split_info']['method']}"
            )

        except Exception as exc:
            failed.append({
                "dataset": dataset_name,
                "error": str(exc),
            })

            print(f"FAILED: {exc}")

    # ------------------------------------------------------
    # Final report
    # ------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL DATASET TEST REPORT")
    print("=" * 65)

    print(f"Total datasets: {len(datasets)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSuccessful datasets:")
        for name in successful:
            print(f"  [OK] {name}")

    if failed:
        print("\nFailed datasets:")
        for item in failed:
            print(f"  [ERROR] {item['dataset']}")
            print(f"          {item['error']}")

    if not failed:
        print("\nAll dataset smoke tests passed.")
    else:
        print("\nSome datasets require further investigation.")
