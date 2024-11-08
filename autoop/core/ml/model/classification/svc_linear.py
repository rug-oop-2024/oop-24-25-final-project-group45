from autoop.core.ml.model import Model
from sklearn.svm import LinearSVC
import numpy as np


# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.predict

class SVCLinear(Model):
    def __init__(self):
        super().__init__()
        self._model = None

    def fit(
            self,
            observations: np.ndarray,
            ground_truth: np.ndarray
    ) -> None:
        """
        Fits the model to the observations and ground truth data.

        Args:
            observations (np.ndarray): X values used to train model
            ground_truth (np.ndarray): True values used to improve model
        """
        self._check_fit_requirements(observations, ground_truth)
        self._model = LinearSVC()
        self._model.fit(observations, ground_truth)
        params = self._model.get_params()
        self._parameters = params

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts observations based on the fitted model.

        Args:
            observations (np.ndarray): Observations matrix for prediction.

        Returns:
            np.ndarray: Predicted values for each observation in X.
        """
        predictions = self._model.predict(observations)
        return predictions
