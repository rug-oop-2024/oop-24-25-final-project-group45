from abc import ABC, abstractmethod
import numpy as np

# Define metric categories for regression and classification
REGRESSION_METRICS = ["MSE", "MAE", "R2"]
CLASSIFICATION_METRICS = ["Accuracy", "Precision", "Recall"]


def get_metric(metric_name: str) -> "Metric":
    """Retrieve the metric instance by its name.

    Args:
        metric_name (str): Name of the metric to retrieve.

    Returns:
        Metric: Instance of the requested metric.

    Raises:
        ValueError: If the provided metric name is invalid.
    """
    available_metrics = REGRESSION_METRICS + CLASSIFICATION_METRICS
    if metric_name not in available_metrics:
        raise ValueError(
            f"'{metric_name}' is not recognized. Available metrics:\n"
            f"{', '.join(available_metrics)}"
        )

    if metric_name == "MSE":
        return MeanSquaredError()
    elif metric_name == "MAE":
        return MeanAbsoluteError()
    elif metric_name == "R2":
        return RSquared()
    elif metric_name == "Accuracy":
        return Accuracy()
    elif metric_name == "Precision":
        return Precision()
    elif metric_name == "Recall":
        return Recall()

    raise ValueError(f"No metric found for: '{metric_name}'.")


class Metric(ABC):
    """Abstract base for defining metrics."""

    def __call__(self, predictions: np.ndarray,
                 actual_values: np.ndarray) -> float:
        """Compute the metric on predictions.

        Args:
            predictions (np.ndarray): Array of predicted values.
            actual_values (np.ndarray): Array of true values.

        Returns:
            float: Calculated metric value.
        """
        return self.compute(predictions, actual_values)

    @abstractmethod
    def compute(self, predictions: np.ndarray,
                actual_values: np.ndarray) -> float:
        """Compute the specific metric on predictions."""
        pass

    def _validate_inputs(self, predictions: np.ndarray,
                         actual_values: np.ndarray) -> None:
        """Validate that prediction and actual values arrays match in length.

        Args:
            predictions (np.ndarray): Predictions.
            actual_values (np.ndarray): True values.

        Raises:
            ValueError: If arrays differ in length or are empty.
        """
        if len(predictions) != len(actual_values):
            raise ValueError(
                f"Mismatch in lengths: predictions ({len(predictions)}) "
                f"vs actual values ({len(actual_values)})."
            )
        if len(predictions) == 0:
            raise ValueError("Predictions and actual values cannot be empty.")


class MeanSquaredError(Metric):
    """Metric for Mean Squared Error in regression."""

    def compute(self, predictions: np.ndarray,
                actual_values: np.ndarray) -> float:
        """Compute MSE by averaging squared differences."""
        self._validate_inputs(predictions, actual_values)
        return float(np.mean((actual_values - predictions) ** 2))


class MeanAbsoluteError(Metric):
    """Metric for Mean Absolute Error in regression."""

    def compute(self, predictions: np.ndarray,
                actual_values: np.ndarray) -> float:
        """Compute MAE by averaging absolute differences."""
        self._validate_inputs(predictions, actual_values)
        return float(np.mean(np.abs(predictions - actual_values)))


class RSquared(Metric):
    """Metric for R-squared in regression."""

    def compute(self, predictions: np.ndarray,
                actual_values: np.ndarray) -> float:
        """Calculate R^2, representing explained variance."""
        self._validate_inputs(predictions, actual_values)
        ss_residuals = np.sum((actual_values - predictions) ** 2)
        ss_total = np.sum((actual_values - np.mean(actual_values)) ** 2)
        return float(1 - ss_residuals / ss_total) \
            if ss_total != 0 else float("nan")


class Accuracy(Metric):
    """Metric for Accuracy in classification."""

    def compute(self, predictions: np.ndarray,
                actual_values: np.ndarray) -> float:
        """Compute accuracy as the fraction of correct predictions."""
        self._validate_inputs(predictions, actual_values)

        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        if actual_values.ndim > 1:
            actual_values = np.argmax(actual_values, axis=1)

        if predictions.shape != actual_values.shape:
            raise ValueError(
                f"Incompatible shapes for accuracy computation: predictions "
                f"{predictions.shape}, actual_values {actual_values.shape}"
            )

        correct_preds = np.sum(predictions == actual_values)
        return float(correct_preds / len(predictions))


class Precision(Metric):
    """Metric for Precision in classification."""

    def compute(self, predictions: np.ndarray,
                actual_values: np.ndarray) -> float:
        """Compute precision as true positives over total positives."""
        self._validate_inputs(predictions, actual_values)
        if actual_values.ndim > 1:
            actual_values = np.argmax(actual_values, axis=1)
        unique_labels = np.unique(actual_values)
        return float(np.mean([self._precision_for_label
                              (predictions, actual_values, label)
                              for label in unique_labels]))

    def _precision_for_label(self, predictions: np.ndarray,
                             actual_values: np.ndarray, label: int) -> float:
        """Helper to calculate precision for a specific label."""
        tp = np.sum((predictions == label) & (actual_values == label))
        fp = np.sum((predictions == label) & (actual_values != label))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


class Recall(Metric):
    """Metric for Recall in classification."""

    def compute(self, predictions: np.ndarray,
                actual_values: np.ndarray) -> float:
        """Compute recall as true positives over total actual positives."""
        self._validate_inputs(predictions, actual_values)
        if actual_values.ndim > 1:
            actual_values = np.argmax(actual_values, axis=1)
        unique_labels = np.unique(actual_values)
        return float(np.mean([self._recall_for_label
                              (predictions, actual_values, label)
                              for label in unique_labels]))

    def _recall_for_label(self, predictions: np.ndarray,
                          actual_values: np.ndarray, label: int) -> float:
        """Helper to calculate recall for a specific label."""
        tp = np.sum((predictions == label) & (actual_values == label))
        fn = np.sum((predictions != label) & (actual_values == label))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
