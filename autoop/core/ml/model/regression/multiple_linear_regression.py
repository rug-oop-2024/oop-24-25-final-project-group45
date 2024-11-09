from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(Model):
    """Multiple Linear Regression model using sklearn library"""

    def __init__(self, *args, **kwargs):
        """Initialize object with optional parameters."""
        super().__init__()
        self._model = LinearRegression(*args, **kwargs)

        # Set hyperparameters using the model's parameter dictionary
        self.parameters = self._model.get_params()
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the regression model with observations and ground truths.

        Args:
            observations (np.ndarray): Observations matrix (samples x variables).
            ground_truth (np.ndarray): Ground truths vector corresponding to X (samples).
        """
        self._check_fit_requirements(observations, ground_truth)

        # Train the model on the provided data
        self._model.fit(observations, ground_truth)

        # Update parameters with coefficients and intercept
        self.parameters.update({
            "coefficients": np.array(self._model.coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
        })

        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the given observations.

        Args:
            X (np.ndarray): Observations matrix for prediction.

        Returns:
            np.ndarray: Predicted values for each observation in X.
        """
        self._check_predict_requirements(X)
        return self._model.predict(X)
