from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier


class KNearestNeighbors(Model):
    """K-Nearest Neighbors using sklearn library."""

    def __init__(self, *args, k_value: int = 3, **kwargs):
        """Initialize knn object.

        Args:
            k_value (int): Neighbors k to consider.
        """
        super().__init__()
        self.k = k_value
        self._model = SkKNeighborsClassifier(
            *args, n_neighbors=self.k, **kwargs
        )

        # Set model parameters
        self.parameters = self._model.get_params()
        self.type = "classification"

    @property
    def k(self) -> int:
        """Retrieve the current value of k."""
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        """Set the value of k with validation."""
        self._k = self._validate_k(value)

    def _validate_k(self, k_value: int) -> int:
        """Check that k is a valid integer greater than 0.

        Args:
            k_value (int): Value of k to validate.

        Returns:
            int: Validated k value.

        Raises:
            TypeError: If k_value is not an integer.
            ValueError: If k_value is not positive.
        """
        if not isinstance(k_value, int):
            raise TypeError("k must be an integer")
        if k_value <= 0:
            raise ValueError("k must be greater than 0")
        return k_value

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the KNN model using observations and ground truths.

        Args:
            X (np.ndarray): Training data, with samples as rows.
            y (np.ndarray): Target labels for each sample in X.
        """
        self._check_fit_requirements(X, y)

        if self.k > X.shape[0]:
            raise ValueError(
                f"k ({self.k}) cannot exceed the number of samples ({X.shape[0]})."
            )

        if y.ndim > 1:
            y = np.argmax(y, axis=1)

        self._model.fit(X, y)
        self._fitted = True
        self._n_features = X.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the given observations.

        Args:
            X (np.ndarray): Data samples to classify.

        Returns:
            np.ndarray: Predicted class labels.
        """
        self._check_predict_requirements(X)
        return self._model.predict(X)
