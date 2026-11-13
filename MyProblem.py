
import numpy as np
from mealpy import Problem


from Config import (
    CLASSIFIER,
    DRAW_CONFUSION_MATRIX,
    AVERAGE_METRIC,
    RANDOM_STATE,
)

from Evaluator import Evaluator


class MyProblem(Problem):
    """
    Binary Feature Selection Problem.

    Training:
        Used to train the classifier.

    Validation:
        Used by the optimizer to calculate the fitness.

    Test:
        Reserved for final evaluation after optimization.

    Fitness:
        alpha * (1 - validation_balanced_accuracy)
        + feature_weight * (selected_features / total_features)
    """

    def __init__(
        self,
        bounds,
        data,
        minmax="min",
        name="FeatureSelection",
        log_to=None,
        classifier=None,
        average_metric=None,
        random_state=None,
        alpha=0.8,
        feature_weight=0.2,
        knn_neighbors=5,
        **kwargs,
    ):
        # --------------------------------------------------
        # 1. Validate and prepare input data
        # --------------------------------------------------
        required_keys = [
            "X_train",
            "y_train",
            "X_validation",
            "y_validation",
            "X_test",
            "y_test",
        ]

        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary.")

        missing_keys = [
            key for key in required_keys if key not in data
        ]

        if missing_keys:
            raise ValueError(
                f"Missing data keys: {missing_keys}"
            )

        self.X_train = np.asarray(data["X_train"])
        self.y_train = np.asarray(data["y_train"]).ravel()

        self.X_validation = np.asarray(data["X_validation"])
        self.y_validation = np.asarray(
            data["y_validation"]
        ).ravel()

        self.X_test = np.asarray(data["X_test"])
        self.y_test = np.asarray(data["y_test"]).ravel()

        # Set feature count before data validation
        if self.X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")

        self.n_features = self.X_train.shape[1]

        self._validate_data()

        # --------------------------------------------------
        # 2. Configuration
        # --------------------------------------------------
        self.classifier = (
            classifier if classifier is not None else CLASSIFIER
        )

        self.average_metric = (
            average_metric
            if average_metric is not None
            else AVERAGE_METRIC
        )

        self.random_state = (
            random_state
            if random_state is not None
            else RANDOM_STATE
        )

        self.alpha = float(alpha)
        self.feature_weight = float(feature_weight)
        self.knn_neighbors = int(knn_neighbors)

        if self.alpha < 0 or self.feature_weight < 0:
            raise ValueError(
                "Fitness weights cannot be negative."
            )

        if not np.isclose(
            self.alpha + self.feature_weight, 1.0
        ):
            raise ValueError(
                "alpha + feature_weight must equal 1."
            )

        if self.knn_neighbors < 1:
            raise ValueError(
                "knn_neighbors must be at least 1."
            )

        # Confusion matrices must not be drawn during optimization.
        self.draw_confusion_matrix = False

        # Retain evaluation history for compatibility and analysis.
        self.eval_stats = []

        # Store the best binary mask evaluated so far.
        self.best_mask = None
        self.best_fitness = np.inf

        # --------------------------------------------------
        # 3. Initialize Mealpy Problem
        # --------------------------------------------------
        super().__init__(
            bounds=bounds,
            minmax=minmax,
            name=name,
            log_to=log_to,
            **kwargs,
        )

        if self.n_dims != self.n_features:
            raise ValueError(
                f"Problem dimension ({self.n_dims}) does not match "
                f"dataset feature count ({self.n_features})."
            )

    # ======================================================
    # Data validation
    # ======================================================
    def _validate_data(self):
        datasets = [
            ("X_train", self.X_train, self.y_train),
            (
                "X_validation",
                self.X_validation,
                self.y_validation,
            ),
            ("X_test", self.X_test, self.y_test),
        ]

        n_features = self.X_train.shape[1]

        if self.X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")

        for name, X, y in datasets:
            if X.ndim != 2:
                raise ValueError(f"{name} must be a 2D array.")

            if X.shape[1] != n_features:
                raise ValueError(
                    f"{name} has an inconsistent number of features."
                )

            if X.shape[0] != len(y):
                raise ValueError(
                    f"Number of samples in {name} and its labels do not match."
                )

            if X.shape[0] == 0:
                raise ValueError(
                    f"{name} cannot be empty."
                )

            if not np.isfinite(X).all():
                raise ValueError(
                    f"{name} contains NaN or infinite values."
                )

        if self.n_features < 1:
            raise ValueError(
                "The dataset must contain at least one feature."
            )

        if len(np.unique(self.y_train)) < 2:
            raise ValueError(
                "Training data must contain at least two classes."
            )

        train_classes = set(np.unique(self.y_train))

        for name, y in [
            ("Validation", self.y_validation),
            ("Test", self.y_test),
        ]:
            unknown_classes = set(np.unique(y)) - train_classes

            if unknown_classes:
                raise ValueError(
                    f"{name} contains classes absent from Training: "
                    f"{unknown_classes}"
                )

    # ======================================================
    # Binary solution validation
    # ======================================================
    def _validate_solution(self, solution):
        """
        Accept only a one-dimensional binary feature mask.

        Valid values: 0 and 1 (including Boolean values).
        No automatic binarization or modification is performed.
        """
        mask = np.asarray(solution)

        if mask.ndim != 1:
            raise ValueError(
                "Solution must be a one-dimensional array."
            )

        if len(mask) != self.n_features:
            raise ValueError(
                f"Solution length ({len(mask)}) does not match "
                f"feature count ({self.n_features})."
            )

        if not np.issubdtype(mask.dtype, np.number) and (
            mask.dtype != np.bool_
        ):
            raise TypeError(
                "Solution must contain numeric or Boolean values."
            )

        if not np.isfinite(mask).all():
            raise ValueError(
                "Solution contains NaN or infinite values."
            )

        if not np.isin(mask, [0, 1]).all():
            raise ValueError(
                "Solution must be strictly binary (0 or 1)."
            )

        return mask.astype(bool, copy=True)

    # ======================================================
    # Fitness calculation
    # ======================================================
    def _calculate_fitness(self, mask):
        """
        Calculate fitness using Validation Balanced Accuracy.

        The classifier is trained only on Training data.
        Validation data is used only for fitness evaluation.
        """
        selected_count = int(np.count_nonzero(mask))
        feature_ratio = selected_count / self.n_features

        # An empty feature subset receives the maximum penalty.
        # The original mask is never modified.
        if selected_count == 0:
            return {
                "accuracy": np.nan,
                "balanced_accuracy": 0.0,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "n_features": 0,
                "feature_ratio": 0.0,
                "classification_error": 1.0,
                "fitness_value": 1.0,
            }

        X_train_selected = self.X_train[:, mask]
        X_validation_selected = self.X_validation[:, mask]

        evaluator = Evaluator(
            train_X=X_train_selected,
            eval_X=X_validation_selected,
            train_Y=self.y_train,
            eval_Y=self.y_validation,
            classifier=self.classifier,
            draw_confusion_matrix=self.draw_confusion_matrix,
            average_metric=self.average_metric,
            random_state=self.random_state,
            knn_neighbors=self.knn_neighbors,
        )

        metrics = evaluator.get_metrics()

        balanced_accuracy = float(
            metrics["balanced_accuracy"]
        )

        classification_error = 1.0 - balanced_accuracy

        fitness = (
            self.alpha * classification_error
            + self.feature_weight * feature_ratio
        )

        return {
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": balanced_accuracy,
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "n_features": selected_count,
            "feature_ratio": feature_ratio,
            "classification_error": classification_error,
            "fitness_value": float(fitness),
        }

    # ======================================================
    # Mealpy objective function
    # ======================================================
    def obj_func(self, solution):
        """
        Objective function called by Mealpy.

        Returns a single scalar fitness value.
        """
        mask = self._validate_solution(solution)

        result = self._calculate_fitness(mask)

        self.eval_stats.append(result.copy())

        fitness = result["fitness_value"]

        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_mask = mask.copy()

        return fitness

    # ======================================================
    # Evaluate a solution on Validation
    # ======================================================
    def evaluate_solution(self, solution):
        """
        Evaluate a binary mask on Validation.

        Returns:
            balanced_accuracy,
            n_selected_features,
            classification_error,
            fitness
        """
        mask = self._validate_solution(solution)

        result = self._calculate_fitness(mask)

        return (
            result["balanced_accuracy"],
            result["n_features"],
            result["classification_error"],
            result["fitness_value"],
        )

    # ======================================================
    # Final evaluation on Test
    # ======================================================
    def evaluate_test(self, solution):
        """
        Final evaluation of a selected feature subset.

        The classifier is retrained on Training + Validation
        and evaluated on the untouched Test set.

        Call this method only after optimization is completed.
        """
        mask = self._validate_solution(solution)

        selected_count = int(np.count_nonzero(mask))

        if selected_count == 0:
            raise ValueError(
                "Cannot evaluate an empty feature subset."
            )

        # Combine Training and Validation for final model fitting.
        X_train_validation = np.concatenate(
            [self.X_train, self.X_validation],
            axis=0,
        )

        y_train_validation = np.concatenate(
            [self.y_train, self.y_validation],
            axis=0,
        )

        X_train_validation_selected = (
            X_train_validation[:, mask]
        )

        X_test_selected = self.X_test[:, mask]

        evaluator = Evaluator(
            train_X=X_train_validation_selected,
            eval_X=X_test_selected,
            train_Y=y_train_validation,
            eval_Y=self.y_test,
            classifier=self.classifier,
            draw_confusion_matrix=DRAW_CONFUSION_MATRIX,
            average_metric=self.average_metric,
            random_state=self.random_state,
            knn_neighbors=self.knn_neighbors,
        )

        metrics = evaluator.get_metrics()

        return {
            **metrics,
            "n_features": selected_count,
            "total_features": self.n_features,
            "feature_ratio": selected_count / self.n_features,
            "selected_mask": mask.astype(int).tolist(),
        }


if __name__ == "__main__":
    from mealpy import BinaryVar

    print("\nMyProblem Smoke Test")
    print("-" * 45)

    rng = np.random.default_rng(42)

    n_samples = 200
    n_features = 10

    X = rng.random((n_samples, n_features))
    y = (
        (X[:, 0] + X[:, 1] + rng.normal(0, 0.2, n_samples)) > 1
    ).astype(int)

    data = {
        "X_train": X[:120],
        "y_train": y[:120],
        "X_validation": X[120:160],
        "y_validation": y[120:160],
        "X_test": X[160:],
        "y_test": y[160:],
    }

    problem = MyProblem(
        bounds=BinaryVar(n_vars=n_features),
        data=data,
        name="SmokeTest",
    )

    print(f"Problem dimensions: {problem.n_dims}")

    # Test 1: Valid binary solution
    mask = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])

    fitness = problem.obj_func(mask)

    assert np.isfinite(fitness)
    assert 0 <= fitness <= 1

    print(f"Valid mask fitness: {fitness:.6f}")

    # Test 2: Empty mask
    empty_mask = np.zeros(n_features, dtype=int)

    empty_fitness = problem.obj_func(empty_mask)

    assert empty_fitness == 1.0
    assert np.all(empty_mask == 0)

    print(f"Empty mask fitness: {empty_fitness:.6f}")

    # Test 3: Invalid non-binary solution
    try:
        problem.obj_func(
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0.5])
        )
        raise AssertionError("Non-binary solution was accepted.")

    except ValueError:
        print("Non-binary solution correctly rejected.")

    # Test 4: Final test evaluation
    test_results = problem.evaluate_test(mask)

    print("\nFinal Test Results:")
    for key, value in test_results.items():
        if key != "selected_mask":
            print(f"{key}: {value}")

    print("\nAll smoke tests passed.")
