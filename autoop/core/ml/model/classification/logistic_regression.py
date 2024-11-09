import numpy as np
from sklearn.linear_model import LogisticRegression

from autoop.core.ml.model.model import Model


class LogisticRegressor(Model):
    """A Logistic Regression implementation of the Model class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the Logistic Regression model with the provided parameters.

        Args:
            *args: Positional arguments for Logistic Regression parameters.
            **kwargs: Keyword arguments for Logistic Regression parameters.
        """
        super().__init__()
        self._model = LogisticRegression(*args, **kwargs)
        self.parameters = self._model.get_params()
        self.type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Train the Logistic Regression model using observations and ground truths.

        Args:
            observations (np.ndarray): Observations used to train the model.
                Row dimension is samples, column dimension is variables.
            ground_truths (np.ndarray): Ground truths corresponding to the
                observations used to train the model. Row dimension is samples.
        """
        if ground_truths.ndim > 1:
            ground_truths = np.argmax(ground_truths, axis=1)

        self._model.fit(observations, ground_truths)

        self.parameters = {
            "coefficients": np.array(self._model.coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
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
