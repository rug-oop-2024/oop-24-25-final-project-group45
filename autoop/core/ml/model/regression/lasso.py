import numpy as np
from sklearn.linear_model import Lasso as SkLasso
from autoop.core.ml.model.model import Model


class Lasso(Model):
    """Lasso regression model as an implementation of the base Model class."""

    def __init__(self, *args, alpha=0.01, **kwargs) -> None:
        """Initialize the Lasso regression model with specified parameters.

        Args:
            alpha (float): Regularization strength; must be a positive float.
            *args: Additional positional arguments for the Lasso model.
            **kwargs: Additional keyword arguments for model configuration.
        """
        super().__init__()
        self._model = SkLasso(*args, alpha=alpha, **kwargs)
        self.parameters = self._model.get_params()
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Train the Lasso model with input data and
        corresponding target values.

        Args:
            observations (np.ndarray): Array of training samples with
            features as columns.
            ground_truth (np.ndarray): Array of target values for each
            training sample.

        Updates:
            self.parameters: Stores fitted coefficients and
            intercept after training.
        """
        self._model.fit(observations, ground_truth)
        self.parameters = {
            "coefficients": np.array(self._model.coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
        }
        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Generate predictions for new input data based on the trained model.

        Args:
            observations (np.ndarray): Array of samples needing predictions,
            with features as columns.

        Returns:
            np.ndarray: Predicted values for each sample, in a
            column vector format.
        """
        return self._model.predict(observations).reshape(-1, 1)
