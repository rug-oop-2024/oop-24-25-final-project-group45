from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression

import numpy as np

class MultinomialLogisticRegression(Model):
    def __init__(self):
        super().__init__()
        self._model = None

    def fit(
            self,
            observations: np.ndarray,
            ground_truth: np.ndarray
    ) -> None:
        """Fit the regression model with observations and ground truths.

        Args:
            observations (np.ndarray): Observations matrix (samples x variables).
            ground_truth (np.ndarray): Ground truths vector corresponding to X (samples).
        """
        self._check_fit_requirements(observations, ground_truth)
        self._model = LogisticRegression(multi_class="auto")
        self._model.fit(observations, ground_truth)
        self.parameters = self._model.get_params()

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Generate predictions for the given observations.

        Args:
            observations (np.ndarray): Observations matrix for prediction.

        Returns:
            np.ndarray: Predicted values for each observation in X.
        """
        self._check_predict_requirements(observations)
        predictions = self._model.predict(observations)
        return predictions
