import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC

from autoop.core.ml.model.model import Model


class DecisionTreeModel(Model):
    """A Decision Tree Classifier implementation of the Model class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the Decision Tree model with the provided parameters.

        Args:
            *args: Positional arguments for Decision Tree parameters.
            **kwargs: Keyword arguments for Decision Tree parameters.
        """
        super().__init__()
        self._model = DTC(*args, **kwargs)
        self.parameters = self._model.get_params()
        self.type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Train the Decision Tree model using observations and ground truths.

        Args:
            observations (np.ndarray): Observations used to train the model.
                Row dimension is samples, column dimension is variables.
            ground_truths (np.ndarray): Ground truths corresponding to the
                observations used to train the model. Row dimension is samples.
        """
        # Convert one-hot-encoded labels to single label indices if necessary
        if ground_truths.ndim > 1:
            ground_truths = np.argmax(ground_truths, axis=1)

        # Train the model
        self._model.fit(observations, ground_truths)

        # Update parameters with feature importances after training
        self.parameters = {
            "feature_importances": np.array(self._model.feature_importances_),
            "max_depth": self._model.tree_.max_depth,
            "n_leaves": self._model.get_n_leaves(),
        }

        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict classes for the given observations.

        Args:
            observations (np.ndarray): The observations for which to predict classes.
                Row dimension is samples, column dimension is variables.

        Returns:
            np.ndarray: Predicted classes for each observation.
        """
        return self._model.predict(observations)

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given observations.

        Args:
            observations (np.ndarray): The observations for which to predict probabilities.
                Row dimension is samples, column dimension is variables.

        Returns:
            np.ndarray: Predicted probabilities for each class in each observation.
        """
        return self._model.predict_proba(observations)
