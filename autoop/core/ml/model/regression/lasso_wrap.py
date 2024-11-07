import numpy as np
from autoop.core.ml.model import Model
from sklearn.linear_model import Lasso


class LassoWrapper(Model):
    """
    Wrapper for sklearn's Lasso model, inheriting from the base class as the other models.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = Lasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Fit the regression model with observations and ground truths.

        Args:
            observations (np.ndarray): Observations matrix (samples x variables).
            ground_truth (np.ndarray): Ground truths vector corresponding to X (samples).
        """
        super()._validate_input(observations, ground_truth)

        self._model.fit(observations, ground_truth)
        self._parameters["coefficients"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Generate predictions for the given observations.

        Args:
            observations (np.ndarray): Observations matrix for prediction.

        Returns:
            np.ndarray: Predicted values for each observation in X.
        """
        super()._validate_num_features(observations)
        predictions = self._model.predict(observations)
        predictions = predictions.reshape(predictions.shape[0], 1).round(2)
        return predictions
