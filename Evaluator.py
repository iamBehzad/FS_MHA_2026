
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class Evaluator:
    """
    Evaluate a classifier using training and evaluation datasets.

    Supported classifiers:
        - KNN
        - RF
        - SVM
        - MLP

    Supported evaluation metrics:
        - Accuracy
        - Balanced Accuracy
        - Precision
        - Recall
        - F1-score

    The classifier is trained only on the training dataset.
    Evaluation is performed on the supplied validation or test dataset.
    """

    SUPPORTED_CLASSIFIERS = ("knn", "rf", "svm", "mlp")

    def __init__(
        self,
        train_X,
        eval_X,
        train_Y,
        eval_Y,
        classifier="knn",
        draw_confusion_matrix=False,
        average_metric="weighted",
        random_state=42,
        knn_neighbors=5
    ):

        self.X_train = np.asarray(train_X)
        self.X_eval = np.asarray(eval_X)

        self.y_train = np.asarray(train_Y).ravel()
        self.y_eval = np.asarray(eval_Y).ravel()

        self.classifier = str(classifier).lower()
        self.draw_confusion_matrix = draw_confusion_matrix
        self.average_metric = average_metric

        self.random_state = random_state
        self.knn_neighbors = knn_neighbors

        self._validate_inputs()

        self.clf = self._build_classifier()

    def _validate_inputs(self):
        """Validate input datasets and classifier settings."""

        if self.X_train.ndim != 2 or self.X_eval.ndim != 2:
            raise ValueError("Feature matrices must be two-dimensional.")

        if self.X_train.shape[0] != len(self.y_train):
            raise ValueError(
                "Training features and labels have inconsistent sample counts."
            )

        if self.X_eval.shape[0] != len(self.y_eval):
            raise ValueError(
                "Evaluation features and labels have inconsistent sample counts."
            )

        if self.X_train.shape[1] != self.X_eval.shape[1]:
            raise ValueError(
                "Training and evaluation datasets must have the same number of features."
            )

        if self.X_train.shape[0] == 0 or self.X_eval.shape[0] == 0:
            raise ValueError("Training and evaluation datasets cannot be empty.")

        if self.X_train.shape[1] == 0:
            raise ValueError("At least one feature must be selected.")

        if len(np.unique(self.y_train)) < 2:
            raise ValueError("Training dataset must contain at least two classes.")

        if not set(np.unique(self.y_eval)).issubset(set(np.unique(self.y_train))):
            raise ValueError(
                "Evaluation dataset contains classes absent from training data."
            )

        if self.classifier not in self.SUPPORTED_CLASSIFIERS:
            raise ValueError(
                f"Unsupported classifier: {self.classifier}. "
                f"Supported classifiers: {self.SUPPORTED_CLASSIFIERS}"
            )

        if self.average_metric not in ("binary", "micro", "macro", "weighted"):
            raise ValueError("Invalid average_metric.")

        if (
            isinstance(self.knn_neighbors, bool)
            or not isinstance(self.knn_neighbors, (int, np.integer))
            or self.knn_neighbors < 1
        ):
            raise ValueError("knn_neighbors must be a positive integer.")

        if self.classifier == "knn" and self.knn_neighbors > len(self.y_train):
            raise ValueError(
                "knn_neighbors cannot exceed the number of training samples."
            )

    def _build_classifier(self):
        """Create the requested classifier."""

        if self.classifier == "knn":
            return KNN(
                n_neighbors=self.knn_neighbors
            )

        elif self.classifier == "rf":
            return RF(
                random_state=self.random_state,
                n_jobs=1
            )

        elif self.classifier == "svm":
            return SVM()

        elif self.classifier == "mlp":
            return MLPClassifier(
                random_state=self.random_state,
                max_iter=500
            )

    def get_metrics(self):
        """
        Train the classifier and calculate evaluation metrics.

        Returns:
            dict: Evaluation metrics.
        """

        self.clf.fit(self.X_train, self.y_train)

        y_pred = self.clf.predict(self.X_eval)

        accuracy = accuracy_score(self.y_eval, y_pred)

        balanced_accuracy = balanced_accuracy_score(
            self.y_eval,
            y_pred
        )

        precision = precision_score(
            self.y_eval,
            y_pred,
            average=self.average_metric,
            zero_division=0
        )

        recall = recall_score(
            self.y_eval,
            y_pred,
            average=self.average_metric,
            zero_division=0
        )

        f1 = f1_score(
            self.y_eval,
            y_pred,
            average=self.average_metric,
            zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        if self.draw_confusion_matrix:
            self._plot_confusion_matrix(y_pred)

        return metrics

    def _plot_confusion_matrix(self, y_pred):
        """Plot the confusion matrix."""

        cm = confusion_matrix(
            self.y_eval,
            y_pred
        )

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.show()
        plt.close()


if __name__ == "__main__":

    # Simple smoke test with a synthetic multiclass dataset

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.7, 0.2, 0.1],
        random_state=42
    )

    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    evaluator = Evaluator(
        train_X=X_train,
        eval_X=X_eval,
        train_Y=y_train,
        eval_Y=y_eval,
        classifier="knn",
        knn_neighbors=5
    )

    results = evaluator.get_metrics()

    print("\nEvaluator Smoke Test")
    print("-" * 35)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
