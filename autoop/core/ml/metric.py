from abc import ABC, abstractmethod

import numpy as np

REGRESSION_METRICS = [
    "MeanSquaredError",
    "RSquared",
    "MeanAbsoluteError",
]

CLASSIFICATION_METRICS = ["Accuracy", "Precision", "Recall"]


def get_metric(name: str) -> "Metric":
    """Return a metric instance corresponding to the provided name.

    Args:
        name (str): name of the type of evaluation method

    Returns:
        Metric: The evaluation

    """
    match name:
        case "RSquared":
            return RSquared()
        case "MeanSquaredError":
            return MeanSquaredError()
        case "MeanAbsoluteError":
            return MeanAbsoluteError()
        case "Recall":
            return Recall()
        case "Accuracy":
            return Accuracy()
        case "Precision":
            return Precision()



class Metric(ABC):
    """Base class for all metrics."""

    def __call__(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate predictions with the class-defined metric.

        Args:
            predictions (np.ndarray): Predicted values
            ground_truth (np.ndarray): True vales

        Returns:
            float: The value of the evaluated metric.
        """
        return self.evaluate(predictions, ground_truth)

    @abstractmethod
    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model based on the given metric.

        Args:
            predictions (np.ndarray): Predicted values
            ground_truth (np.ndarray): True vales

        Returns:
            float: The value of the evaluated metric.
        """
        pass

    def _check_dimensions(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> None:
        """Check if the predictions and ground_truth have the right dimensions.

        Args:
            predictions (np.ndarray): Predicted values
            ground_truth (np.ndarray): True vales

        Raises:
            ValueError: If the number of predictions does not equal the number
                of ground truth labels.
            ValueError: If there are no predictions or ground_truths.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"The number of predictions ({len(predictions)}) must equal ",
                f"the number of ground truth labels ({len(ground_truth)}).",
            )
        if len(predictions) == 0:
            raise ValueError(
                "Predictions and ground truth arrays cannot be empty."
            )


class MeanAbsoluteError(Metric):
    """Mean absolute error class"""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's mean absolute error.

        Measures the average size of mistakes in a collection of predictions.

        Args:
            predictions (np.ndarray): Predicted values
            ground_truth (np.ndarray): True vales

        Returns:
            float: The mean absolute error of the model.
        """
        self._check_dimensions(predictions, ground_truth)

        absolute_errors = np.abs(predictions - ground_truth)
        mean_absolute_error = np.mean(absolute_errors)
        return float(mean_absolute_error)


class RSquared(Metric):
    """Rsquared class"""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's rsquared value.

        Measures the proportion of variance that can be explained
        by the independent variables.

        Args:
            predictions (np.ndarray): Predicted values
            ground_truth (np.ndarray): True vales

        Returns:
            float: The proportion of variance that can be explained
                by the independent variables of the model between -∞ and 1.
        """
        self._check_dimensions(predictions, ground_truth)

        residual_sum_of_squares = np.sum((ground_truth - predictions) ** 2)
        sum_of_squares_total = np.sum(
            (ground_truth - np.mean(ground_truth)) ** 2
        )
        if sum_of_squares_total == 0:
            return float("nan")

        return float(1 - residual_sum_of_squares / sum_of_squares_total)

class MeanSquaredError(Metric):
    """Mean squared error class"""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Calculate average squared distance from ground_truth

        Args:
            predictions (np.ndarray): Predicted values
            ground_truth (np.ndarray): True vales

        Returns:
            float: The mean squared error of the model
        """
        self._check_dimensions(predictions, ground_truth)

        squared_errors = (ground_truth - predictions) ** 2
        mse = np.mean(squared_errors)
        return float(mse)

class Recall(Metric):
    """Recall class"""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's recall ability.

        Measures the classifier's ability to find all correct predictions for
        each label. Recall = True positive / (True positive + False negative)

        Args:
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            float: The recall of the model between 0 and 1
        """
        self._check_dimensions(predictions, ground_truth)

        if ground_truth.ndim > 1:
            # The ground_truth is one-hot-encoded so get labels
            ground_truth = np.argmax(ground_truth, axis=1)

        unique_labels = np.unique(ground_truth)
        num_unique_labels = len(unique_labels)

        total_recall = 0.0

        for unique_label in unique_labels:
            total_recall += self._calculate_label_recall(
                unique_label, predictions, ground_truth
            )

        return float(total_recall / num_unique_labels)

    def _calculate_label_recall(
        self,
        unique_label: int | str,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Evaluate the model's recall of one label.

        Recall = True positive / (True positive + False negative)

        Args:
            unique_label (int | str): The label for which the models recall
                needs to be calculated.
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            float: The recall of the model between 0 and 1 of one label.
        """
        # Create boolean arrays that indicate matches for the unique label
        # in both predictions and ground truth.
        match_gt = ground_truth == unique_label
        match_pred = predictions == unique_label

        # Count the true positives and false negatives using the arrays.
        tp = np.sum(match_gt & match_pred)
        fn = np.sum(match_gt & ~match_pred)

        # Avoid dividing by zero
        if tp + fn > 0:
            return float(tp / (tp + fn))

        return 0.0


class Precision(Metric):
    """Precision class"""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's precision.

        Measures the accuracy of positive predictions for each label.
        Precision = True positive / (True positive + False positive)

        Args:
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            float: The precision of the model between 0 and 1
        """
        self._check_dimensions(predictions, ground_truth)

        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)

        unique_labels = np.unique(ground_truth)
        num_unique_labels = len(unique_labels)

        total_precision = 0.0

        for unique_label in unique_labels:
            total_precision += self._calculate_label_precision(
                unique_label, predictions, ground_truth
            )

        return float(total_precision / num_unique_labels)

    def _calculate_label_precision(
        self,
        unique_label: int | str,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Evaluate the model's precision of one label.

        Precision = True positive / (True positive + False positive)

        Args:
            unique_label (int | str): The label for which the models precision
                needs to be calculated.
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            float: The precision of the model between 0 and 1 of one label.
        """
        match_gt = ground_truth == unique_label
        match_pred = predictions == unique_label


        tp = np.sum(match_gt & match_pred)
        fp = np.sum(~match_gt & match_pred)

        # Avoid dividing by zero
        if tp + fp > 0:
            return float(tp / (tp + fp))

        return 0.0

class Accuracy(Metric):
    """Accuracy class"""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's accuracy.

        Measures the ratio of correct predictions.
        Number of correct predictions / Total number of predictions.

        Args:
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels.
                (Must match number of predictions)

        Returns:
            float: The accuracy of the model between 0 and 1
        """
        self._check_dimensions(predictions, ground_truth)

        # If ground_truth is one-hot-encoded get the correct labels
        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)

        correct_predictions = np.sum(predictions == ground_truth)
        return float(correct_predictions / len(predictions))

