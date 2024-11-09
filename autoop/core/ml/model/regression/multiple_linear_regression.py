import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(Model):
    """Multiple Linear Regression model using sklearn library."""

    def __init__(self, *args, **kwargs):
        """Initialize object with optional parameters."""
        super().__init__()
        self._model = LinearRegression(*args, **kwargs)

        self.parameters = self._model.get_params()
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the regression model with observations and ground truths.

        Args:
            observations (np.ndarray): Observations matrix
            (samples x variables).
            ground_truth (np.ndarray): Ground truths vector
            corresponding to X (samples).
        """
        self._model.fit(observations, ground_truth)

        self.parameters.update(
            {
                "coefficients": np.array(self._model.coef_),
                "intercept": np.atleast_1d(self._model.intercept_),
            }
        )

        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Generate predictions for the given observations.

        Args:
            observations (np.ndarray): Observations matrix for prediction.

        Returns:
            np.ndarray: Predicted values for each observation.
        """
        return self._model.predict(observations)
