import numpy as np
from sklearn.ensemble import RandomForestClassifier
from autoop.core.ml.model.model import Model


class RandomForest(Model):
    """Random Forest Classifier implementation of the Model class."""

    def __init__(self, n_estimators=100, *args, **kwargs) -> None:
        """Initialize the Random Forest model with specified parameters.

        Args:
            n_estimators (int): Number of trees in the forest.
            *args: Additional positional arguments for the classifier.
            **kwargs: Additional keyword arguments for the classifier.
        """
        super().__init__()
        self._model = (RandomForestClassifier
                       (n_estimators=n_estimators, *args, **kwargs))
        self.parameters = self._model.get_params()
        self.type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Train the Random Forest model using observations and ground truths.

        Args:
            observations (np.ndarray): Features for training data.
            ground_truths (np.ndarray): True labels for training data.
        """
        if ground_truths.ndim > 1:
            ground_truths = np.argmax(ground_truths, axis=1)

        self._model.fit(observations, ground_truths)
        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict classes for the given observations.

        Args:
            observations (np.ndarray): Features of data to predict.

        Returns:
            np.ndarray: Predicted class labels for each observation.
        """
        return self._model.predict(observations)
