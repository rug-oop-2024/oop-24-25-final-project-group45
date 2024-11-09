import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from autoop.core.ml.model.model import Model


class KNearestNeighbors(Model):
    """A custom implementation of the k-Nearest Neighbors (KNN) classifier."""

    def __init__(self, *args, k_value: int = 3, **kwargs) -> None:
        """Initialize the KNN model with a specified number of neighbors.

        Args:
            k_value (int): Number of nearest neighbors to consider.
        """
        super().__init__()
        self.k = k_value
        self._model = KNN(
            *args, n_neighbors=self.k, **kwargs
        )
        self.parameters = self._model.get_params()
        self.type = "classification"

    @property
    def k(self) -> int:
        """Retrieve the current value of k (number of neighbors)."""
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        """Setter for value of k.

        Args:
            value (int): The value for k to set.

        Raises:
            TypeError: If the provided value is not an integer.
            ValueError: If the provided value is less than or equal to zero.
        """
        self._k = self._check_valid_k(value)

    def _check_valid_k(self, k_value: int) -> int:
        """Ensure k is a positive integer.

        Args:
            k_value (int): The intended value for k.

        Returns:
            int: A validated k-value.

        Raises:
            TypeError: If k_value is not an integer.
            ValueError: If k_value is not positive.
        """
        if not isinstance(k_value, int):
            raise TypeError("The number of neighbors, k, must be an integer.")
        if k_value <= 0:
            raise ValueError("The number of neighbors, k, must be "
                             "greater than zero.")
        return k_value

    def fit(self, observations: np.ndarray, targets: np.ndarray) -> None:
        """Train the KNN model on provided data.

        Args:
            observations (np.ndarray): Training data with samples as rows
                and features as columns.
            targets (np.ndarray): Target labels for each observation.

        Raises:
            ValueError: If k exceeds the number of samples.
        """
        if self.k > observations.shape[0]:
            raise ValueError(
                f"The number of neighbors, k ({self.k}), "
                f"cannot exceed the number of samples "
                f"({observations.shape[0]})."
            )

        if targets.ndim > 1:
            targets = np.argmax(targets, axis=1)

        self._model.fit(observations, targets)

        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Generate predictions for new data.

        Args:
            observations (np.ndarray): Input data where rows are samples
                and columns are features.

        Returns:
            np.ndarray: Predicted labels for each observation.
        """
        if not self._fitted:
            raise RuntimeError("The model must be fitted before making "
                               "predictions.")
        return self._model.predict(observations)
