from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.svm import SVR


class SupportVectorRegressor(Model):
    """A Support Vector Regressor (SVR) implementation of the Model class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the SVR model with the provided parameters.

        Args:
            *args: Positional arguments for SVR parameters.
            **kwargs: Keyword arguments for SVR parameters.
        """
        super().__init__()
        self._model = SVR(*args, **kwargs)
        self.parameters = self._model.get_params()
        self.type = "regression"

    def fit(self, observations: np.ndarray, targets: np.ndarray) -> None:
        """Train the SVR model using observations and target values.

        Args:
            observations (np.ndarray): Observations used to train the model.
                Row dimension is samples, column dimension is features.
            targets (np.ndarray): Target values corresponding to the observations.
                Row dimension is samples.
        """
        self._model.fit(observations, targets)

        self.parameters = {
            "support_vectors": np.array(self._model.support_),
            "dual_coefficients": np.array(self._model.dual_coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
        }

        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict values for the given observations.

        Args:
            observations (np.ndarray): Observations for which to predict values.
                Row dimension is samples, column dimension is features.

        Returns:
            np.ndarray: Predicted values for each observation.
        """
        return self._model.predict(observations)
